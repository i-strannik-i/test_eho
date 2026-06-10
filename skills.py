#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Менеджер скиллов Эхо с поддержкой слэш-команд (/).
Потоковый вывод через файл (обход Python 3.14 бага).
Логи перезаписываются при каждом запуске скилла.
"""
import os
import sys
import json
import re
import threading
import subprocess
import time
from datetime import datetime

from project_paths import BUILD_RUN_LOG_FILE, GGUF_MODEL_PATH, KNOWLEDGE_INPUT_DIR, LOGS_DIR, LORA_OUTPUT_DIR, SKILL_LOG_FILE, TRAINING_DATA_FILE, TRAINING_RUN_LOG_FILE

CONTROL_TEST_QUESTIONS = (
    "Что такое Эхо?",
    "Для чего в проекте используется локальная память?",
    "Что делает команда /учиться?",
    "Что делает команда /собрать?",
    "Что такое GGUF-модель?",
    "Что такое LoRA-дообучение?",
    "Что такое checkpoint в обучении?",
    "С какого места продолжится обучение после перезапуска?",
    "Где лежат файлы знаний проекта?",
    "Как проверить, чему уже научилась модель?",
    "Как переключить собранную модель в Эхо?",
    "Что делать, если обучение было прервано?",
)


class SkillManager:
    def __init__(self, assistant):
        self.assistant = assistant
        self.running_skill = None
        self.active_process = None
        self.stop_requested = False
        self.last_run_interrupted = False
        self.skill_log = []
        self.gui_callback = None
        self.log_file = SKILL_LOG_FILE
        self.runtime_state = self._new_runtime_state()
        self._reset_session_logs()
        
        self.skills = {
            "помощь": (self.skill_help, "Список всех команд", True),
            "стат": (self.skill_stats, "Статистика памяти", True),
            "время": (self.skill_time, "Текущее время", True),
            "законы": (self.skill_laws, "Этические законы", True),
            "проверить": (self.skill_check_learning, "Проверить, чему уже научилась модель", True),
            "тест": (self.skill_run_model_test, "Прогнать контрольные вопросы по модели", True),
            "очистить": (self.skill_clear_chat, "Очистить чат", True),
            "сброс": (self.skill_memory_cleanup, "Сбросить слабые связи", False),
            "модель": (self.skill_switch_model, "Переключить модель", False),
            "cpu": (self.skill_cpu, "Управление потоками CPU", True),
            "ctx": (self.skill_ctx, "Размер контекстного окна", True),
            "restart": (self.skill_restart, "Перезагрузить модель", True),
            "учить": (self.skill_ingest, "Обработать файлы знаний", False),
            "экспорт": (self.skill_export, "Экспорт знаний в JSON", False),
            "учиться": (self.skill_training, "Запустить дообучение + автосборку", False),
            "собрать": (self.skill_build_gguf, "Собрать GGUF из весов", False),
        }

    def _reset_session_logs(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        for log_path in (self.log_file, TRAINING_RUN_LOG_FILE, BUILD_RUN_LOG_FILE):
            try:
                with open(log_path, "w", encoding="utf-8"):
                    pass
            except OSError:
                pass
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[СКИЛЛ {timestamp}] {message}"
        self.skill_log.append(full_msg)
        self._update_runtime_state(message)
        
        # Дописываем в лог-файл (файл очищается в начале скилла)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} {message}\n")
        except Exception:
            pass
        
        if self.gui_callback:
            try:
                self.gui_callback(full_msg)
            except Exception:
                pass
        print(full_msg)
    
    def register_gui(self, callback):
        self.gui_callback = callback

    def _new_runtime_state(self):
        return {
            "task": None,
            "status_line": "ожидание",
            "current_step": None,
            "total_steps": None,
            "current_epoch": None,
            "total_epochs": None,
            "checkpoint": None,
            "started_at": None,
            "elapsed_seconds": None,
            "elapsed_updated_at": None,
        }

    def _reset_runtime_state(self, task_name=None):
        self.runtime_state = self._new_runtime_state()
        self.runtime_state["task"] = task_name
        self.runtime_state["started_at"] = time.time() if task_name else None

    def _update_runtime_state(self, message):
        state = self.runtime_state

        if message.startswith("Запуск: "):
            state["status_line"] = message.replace("Запуск: ", "", 1)
            return

        if "НАЙДЕН ЧЕКПОИНТ!" in message:
            checkpoint_path = message.split(":", 1)[-1].strip()
            state["checkpoint"] = os.path.basename(checkpoint_path)
            return

        if "ЧЕКПОИНТ НЕ НАЙДЕН" in message:
            state["checkpoint"] = "нет"
            return

        checkpoint_match = re.search(r"checkpoint-\d+", message)
        lowered = message.lower()
        if checkpoint_match and ("сохран" in lowered or "возобнов" in lowered):
            state["checkpoint"] = checkpoint_match.group(0)

        checkpoint_field_match = re.search(r"checkpoint:\s*([^,\n]+)", message, re.IGNORECASE)
        if checkpoint_field_match:
            state["checkpoint"] = checkpoint_field_match.group(1).strip()

        time_match = re.search(r"время:\s*(\d+)\s*сек", message, re.IGNORECASE)
        if time_match:
            state["elapsed_seconds"] = int(time_match.group(1))
            state["elapsed_updated_at"] = time.time()

        epoch_match = re.search(r"эпоха\s+(\d+)/(\d+)", message, re.IGNORECASE)
        if epoch_match:
            state["current_epoch"] = int(epoch_match.group(1))
            state["total_epochs"] = int(epoch_match.group(2))

        step_match = re.search(r"шаг\s+(\d+)(?:/|\s+из\s+)(\d+)", message, re.IGNORECASE)
        if step_match:
            state["current_step"] = int(step_match.group(1))
            state["total_steps"] = int(step_match.group(2))

        plan_match = re.search(r"План шагов:\s*(\d+)", message)
        if plan_match:
            state["total_steps"] = int(plan_match.group(1))

        if (
            message.startswith("Статус:")
            or message.startswith("ШАГ ")
            or message.startswith("Шаг ")
            or message.startswith("Эпоха ")
            or message.startswith("Сохранение модели")
            or message.startswith("Обучение завершено")
        ):
            state["status_line"] = message

    def get_runtime_snapshot(self):
        snapshot = dict(self.runtime_state)
        snapshot["active_task"] = self.running_skill
        snapshot["summary"] = self._format_runtime_summary(snapshot)
        return snapshot

    def _format_runtime_summary(self, snapshot):
        active_task = snapshot.get("active_task")
        if not active_task:
            return "Ожидание"

        step = snapshot.get("current_step")
        total_steps = snapshot.get("total_steps")
        epoch = snapshot.get("current_epoch")
        total_epochs = snapshot.get("total_epochs")
        checkpoint = snapshot.get("checkpoint") or "нет"
        started_at = snapshot.get("started_at")
        elapsed = self._resolve_elapsed_seconds(snapshot, started_at)
        time_text = f"{elapsed} сек"

        if step and total_steps and epoch and total_epochs:
            return f"Шаг {step} из {total_steps}, эпоха {epoch}/{total_epochs}, checkpoint: {checkpoint}, время: {time_text}"
        if epoch and total_epochs:
            return f"Эпоха {epoch}/{total_epochs}, checkpoint: {checkpoint}, время: {time_text}"
        status_line = snapshot.get("status_line") or active_task
        return f"{status_line}, checkpoint: {checkpoint}, время: {time_text}"

    def _resolve_elapsed_seconds(self, snapshot, started_at):
        parsed_elapsed = snapshot.get("elapsed_seconds")
        if parsed_elapsed is not None:
            wall_elapsed = int(time.time() - started_at) if started_at else parsed_elapsed
            return max(parsed_elapsed, wall_elapsed)
        if started_at:
            return int(time.time() - started_at)
        return 0
    
    def parse_command(self, text):
        text_stripped = text.strip()
        if not text_stripped:
            return None, None
        if text_stripped.startswith("/"):
            parts = text_stripped[1:].split(maxsplit=1)
            if parts:
                skill_name = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                return skill_name, args
        text_lower = text_stripped.lower()
        prefixes = ["скилл ", "skill ", "скил "]
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                parts = text_stripped[len(prefix):].split(maxsplit=1)
                if parts:
                    skill_name = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    return skill_name, args
        return None, None
    
    def execute(self, skill_name, args="", callback_dict=None):
        if skill_name not in self.skills:
            return f"Неизвестная команда: '/{skill_name}'\nНапишите /помощь для списка команд."
        if self.running_skill:
            return f"Уже запущен навык: {self.running_skill}. Дождитесь завершения."
        func, description, is_instant = self.skills[skill_name]
        if is_instant:
            try:
                result = func(args, callback_dict)
                return result
            except Exception as e:
                return f"Ошибка в команде '/{skill_name}': {e}"
        self.log(f"Запуск: {description}")
        def run():
            try:
                func(args, callback_dict)
                if not self.last_run_interrupted:
                    self.log(f"Команда '/{skill_name}' завершена.")
            except Exception as e:
                self.log(f"Ошибка в '/{skill_name}': {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.active_process = None
                self.stop_requested = False
                self.last_run_interrupted = False
                self.running_skill = None
                self._reset_runtime_state()
        self.running_skill = skill_name
        self._reset_runtime_state(skill_name)
        threading.Thread(target=run, daemon=True).start()
        return f"[СКИЛЛ] Запущен навык '{skill_name}'. Смотри прогресс в логе."

    def interrupt_running_skill(self):
        if not self.running_skill:
            return False, "Сейчас нет активной задачи для прерывания."

        self.stop_requested = True
        process = self.active_process
        if process and process.poll() is None:
            try:
                process.terminate()
                self.log(f"Запрошено прерывание '/{self.running_skill}'...")
                return True, f"Прерываю '/{self.running_skill}'."
            except Exception as e:
                self.log(f"Ошибка прерывания: {e}")
                return False, f"Не удалось прервать '/{self.running_skill}': {e}"

        self.log(f"Запрошено прерывание '/{self.running_skill}', ожидаю безопасного завершения...")
        return True, f"Запрошено завершение '/{self.running_skill}'."
    
    def skill_help(self, args="", callback_dict=None):
        lines = ["Доступные команды Эхо:", "-" * 40]
        categories = {
            "Информация": ["помощь", "стат", "время", "законы", "проверить", "тест"],
            "Управление": ["очистить", "сброс", "модель"],
            "Ресурсы": ["cpu", "ctx", "restart"],
            "Данные": ["учить", "экспорт"],
            "Обучение": ["учиться", "собрать"],
        }
        for category, names in categories.items():
            lines.append(f"\n{category}:")
            for name in names:
                if name in self.skills:
                    _, desc, _ = self.skills[name]
                    lines.append(f"  /{name} - {desc}")
        lines.append("\n" + "-" * 40)
        lines.append("Пример: /стат или /cpu 8")
        return "\n".join(lines)
    
    def skill_stats(self, args="", callback_dict=None):
        with self.assistant.db_lock:
            cursor = self.assistant.cursor
            cursor.execute("SELECT COUNT(*) FROM memory")
            memory_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM logic_laws")
            laws_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM learned_knowledge")
            knowledge_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM risk_flags")
            risks_count = cursor.fetchone()[0]
        topics_count = len(self.assistant.dynamic_topics)
        links_count = len(self.assistant.associative_links)
        lines = [
            "Статистика Эхо:",
            "-" * 40,
            f"  Диалогов: {memory_count}",
            f"  Законов логики: {laws_count}",
            f"  Знаний: {knowledge_count}",
            f"  Флагов риска: {risks_count}",
            f"  Активных тем: {topics_count}",
            f"  Ассоциативных связей: {links_count}",
            "",
            "Ресурсы:",
            f"  Потоков CPU: {self.assistant.cpu_threads}",
            f"  Контекстное окно: {self.assistant.n_ctx} токенов",
            f"  Максимум ядер в системе: {os.cpu_count() or '?'}",
            "",
            "Состояние:",
            f"  Режим: {self.assistant.cognitive_mode}",
            f"  Нейросеть: {'АКТИВНА' if self.assistant.embedding_model else 'ОТКЛЮЧЕНА'}",
            f"  Локальная модель: {'ЗАГРУЖЕНА' if self.assistant.local_brain else 'НЕ ЗАГРУЖЕНА'}",
        ]
        return "\n".join(lines)
    
    def skill_time(self, args="", callback_dict=None):
        now = datetime.now()
        return f"Сейчас: {now.strftime('%H:%M:%S')} ({now.strftime('%d.%m.%Y')})"
    
    def skill_laws(self, args="", callback_dict=None):
        e = self.assistant.ethics
        lines = ["Мои этические законы:", "-" * 40]
        status = "ВКЛЮЧЕНЫ" if e.get("enabled", True) else "ОТКЛЮЧЕНЫ"
        lines.append(f"Статус: {status}\n")
        for key, value in e.items():
            if key != "enabled":
                lines.append(f"  {key.upper()}: {value}")
        return "\n".join(lines)

    def skill_check_learning(self, args="", callback_dict=None):
        question = args.strip()
        if question.lower() in ("тест", "test"):
            return self.skill_run_model_test("", callback_dict)
        if question:
            return self._ask_current_model(question)

        latest_checkpoint = self._latest_checkpoint_dir()
        checkpoint_name = os.path.basename(latest_checkpoint) if latest_checkpoint else "нет"
        dataset_size = self._count_jsonl_rows(TRAINING_DATA_FILE)
        gguf_exists = os.path.exists(GGUF_MODEL_PATH)
        model_loaded = bool(getattr(self.assistant, "local_brain", None))

        lines = [
            "Проверка текущего состояния обучения:",
            "-" * 40,
            f"  Датасет: {TRAINING_DATA_FILE} ({dataset_size} примеров)" if dataset_size is not None else f"  Датасет: {TRAINING_DATA_FILE} (не найден)",
            f"  Последний checkpoint: {checkpoint_name}",
            f"  GGUF-модель: {'есть' if gguf_exists else 'ещё не собрана'}",
            f"  Загружена в Эхо: {'да' if model_loaded else 'нет'}",
            "",
            "Как проверить результат:",
        ]

        if latest_checkpoint:
            lines.append("  Во время обучения прогресс уже сохраняется в checkpoint-*.")
        else:
            lines.append("  Пока checkpoint ещё не создан, значит обучение слишком рано оценивать.")

        if gguf_exists and model_loaded:
            lines.append("  Спросите прямо модель: /проверить что ты знаешь о ...")
            lines.append("  Пример: /проверить что такое Эхо?")
        elif gguf_exists:
            lines.append("  Сначала переключите собранную модель: /модель")
            lines.append("  Потом спросите: /проверить что такое Эхо?")
        else:
            lines.append("  Для проверки ответов после обучения нужна сборка GGUF: /собрать")

        lines.append("")
        lines.append("Что важно:")
        lines.append("  Пока идёт обучение, Эхо в чате не использует LoRA checkpoint напрямую.")
        lines.append("  Проверять реальные ответы удобнее после /собрать и /модель.")
        return "\n".join(lines)

    def skill_run_model_test(self, args="", callback_dict=None):
        local_brain = getattr(self.assistant, "local_brain", None)
        if not local_brain:
            if os.path.exists(GGUF_MODEL_PATH):
                return "Собранная модель есть, но сейчас не загружена. Сначала выполните /модель, потом запустите /тест."
            return "Сейчас загруженной локальной модели нет. Сначала завершите /собрать, затем выполните /модель."
        if self.running_skill:
            return f"Уже запущен навык: {self.running_skill}. Дождитесь завершения."

        self.running_skill = "тест"
        self._reset_runtime_state("тест")
        self.log("Запуск: Контрольный тест модели")

        def run():
            try:
                self._run_control_test_suite()
                if not self.last_run_interrupted:
                    self.log("Команда '/тест' завершена.")
            except Exception as e:
                self.log(f"Ошибка в '/тест': {e}")
            finally:
                self.active_process = None
                self.stop_requested = False
                self.last_run_interrupted = False
                self.running_skill = None
                self._reset_runtime_state()

        threading.Thread(target=run, daemon=True).start()
        return "[СКИЛЛ] Запущен тест модели. Смотри прогресс в логе."
    
    def skill_clear_chat(self, args="", callback_dict=None):
        if callback_dict and callback_dict.get("clear_chat"):
            callback_dict["clear_chat"]()
            return "Чат очищен."
        return "Функция очистки чата недоступна."
    
    def skill_cpu(self, args="", callback_dict=None):
        args = args.strip()
        if args.lower() == "auto":
            max_threads = os.cpu_count() or 4
            optimal = max(2, max_threads - 2)
            old = self.assistant.cpu_threads
            self.assistant.cpu_threads = optimal
            self.assistant.save_config()
            return (
                f"Автонастройка CPU:\n"
                f"  Обнаружено ядер: {max_threads}\n"
                f"  Установлено потоков: {optimal} (было {old})\n"
                f"Выполните /restart чтобы применить."
            )
        if not args:
            current = self.assistant.cpu_threads
            max_threads = os.cpu_count() or 4
            return (
                f"Текущие параметры CPU:\n"
                f"  Потоков: {current}\n"
                f"  Максимум ядер в системе: {max_threads}\n"
                f"  Контекстное окно: {self.assistant.n_ctx}\n\n"
                f"Чтобы изменить: /cpu 8\n"
                f"Автонастройка: /cpu auto\n"
                f"После изменения выполните /restart"
            )
        try:
            new_threads = int(args)
        except ValueError:
            return f"Ошибка: '{args}' не является числом.\nПример: /cpu 8"
        max_threads = os.cpu_count() or 4
        if new_threads < 1:
            return "Минимум 1 поток."
        if new_threads > max_threads:
            return f"У вашей системы только {max_threads} ядер.\nРекомендую: /cpu {max_threads - 2}"
        if new_threads > 32:
            return "Слишком много потоков. Максимум 32."
        old_threads = self.assistant.cpu_threads
        self.assistant.cpu_threads = new_threads
        self.assistant.save_config()
        return (
            f"Потоки CPU изменены: {old_threads} -> {new_threads}\n"
            f"Выполните /restart чтобы применить изменения."
        )
    
    def skill_ctx(self, args="", callback_dict=None):
        args = args.strip()
        if not args:
            return (
                f"Текущее контекстное окно: {self.assistant.n_ctx} токенов\n\n"
                f"Чтобы изменить: /ctx 4096\n"
                f"Большие значения требуют больше RAM."
            )
        try:
            new_ctx = int(args)
        except ValueError:
            return f"Ошибка: '{args}' не является числом.\nПример: /ctx 4096"
        if new_ctx < 512:
            return "Минимум 512 токенов."
        if new_ctx > 32768:
            return "Максимум 32768 токенов."
        old_ctx = self.assistant.n_ctx
        self.assistant.n_ctx = new_ctx
        self.assistant.save_config()
        return (
            f"Контекстное окно изменено: {old_ctx} -> {new_ctx}\n"
            f"Выполните /restart чтобы применить."
        )
    
    def skill_restart(self, args="", callback_dict=None):
        self.log("Перезагрузка модели...")
        model_path = GGUF_MODEL_PATH
        if not os.path.exists(model_path):
            return f"Модель не найдена: {model_path}\nСначала выполните /собрать"
        try:
            if self.assistant.local_brain:
                del self.assistant.local_brain
                self.assistant.local_brain = None
                import gc
                gc.collect()
                self.log("Старая модель выгружена из памяти.")
            from llama_cpp import Llama
            self.assistant.local_brain = Llama(
                model_path=model_path,
                n_ctx=self.assistant.n_ctx,
                n_threads=self.assistant.cpu_threads,
                n_gpu_layers=0
            )
            self.log(f"Модель перезагружена!")
            self.log(f"  Потоков CPU: {self.assistant.cpu_threads}")
            self.log(f"  Контекст: {self.assistant.n_ctx}")
            return (
                f"Модель перезагружена с новыми параметрами:\n"
                f"  Потоков CPU: {self.assistant.cpu_threads}\n"
                f"  Контекстное окно: {self.assistant.n_ctx}"
            )
        except Exception as e:
            self.log(f"Ошибка перезагрузки: {e}")
            return f"Не удалось перезагрузить модель: {e}"
    
    def skill_ingest(self, args="", callback_dict=None):
        self.log(f"Обработка файлов из {KNOWLEDGE_INPUT_DIR}...")
        with self.assistant.db_lock:
            count = self.assistant.process_knowledge_inbox()
        if count > 0:
            self.log(f"Усвоено файлов: {count}")
        else:
            self.log(f"Папка {KNOWLEDGE_INPUT_DIR} пуста.")
    
    def skill_memory_cleanup(self, args="", callback_dict=None):
        self.log("Очистка слабых воспоминаний...")
        with self.assistant.db_lock:
            cursor = self.assistant.cursor
            cursor.execute("DELETE FROM memory WHERE weight < 0.5 AND uses = 0")
            deleted_memory = cursor.rowcount
            cursor.execute("DELETE FROM learned_knowledge WHERE weight < 0.5 AND uses = 0")
            deleted_knowledge = cursor.rowcount
            self.assistant.conn.commit()
            self.assistant.cleanup_memory()
        self.log(f"Удалено диалогов: {deleted_memory}")
        self.log(f"Удалено знаний: {deleted_knowledge}")
    
    def skill_switch_model(self, args="", callback_dict=None):
        self.log("Переключение модели...")
        gguf_path = GGUF_MODEL_PATH
        if not os.path.exists(gguf_path):
            self.log(f"Модель не найдена: {gguf_path}")
            self.log("Сначала запустите /собрать.")
            return
        try:
            from llama_cpp import Llama
            if self.assistant.local_brain:
                del self.assistant.local_brain
            self.assistant.local_brain = Llama(
                model_path=gguf_path,
                n_ctx=self.assistant.n_ctx,
                n_threads=self.assistant.cpu_threads
            )
            self.log("Модель переключена на дообученную!")
        except Exception as e:
            self.log(f"Ошибка загрузки: {e}")
    
    def skill_export(self, args="", callback_dict=None):
        self.log("Экспорт знаний в JSON...")
        export_data = {
            "exported_at": str(datetime.now()),
            "memory": [], "logic_laws": [], "learned_knowledge": [],
            "topics": self.assistant.dynamic_topics,
        }
        with self.assistant.db_lock:
            cursor = self.assistant.cursor
            cursor.execute("SELECT user_text, answer, weight FROM memory")
            export_data["memory"] = [{"user": r[0], "answer": r[1], "weight": r[2]} for r in cursor.fetchall()]
            cursor.execute("SELECT premise, consequence, solution, weight FROM logic_laws")
            export_data["logic_laws"] = [{"premise": r[0], "consequence": r[1], "solution": r[2], "weight": r[3]} for r in cursor.fetchall()]
            cursor.execute("SELECT content, knowledge_type, weight FROM learned_knowledge")
            export_data["learned_knowledge"] = [{"content": r[0], "type": r[1], "weight": r[2]} for r in cursor.fetchall()]
        filename = f"echo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        self.log(f"Экспорт сохранён: {filename}")
    
    # =========================================================================
    # ПОТОКОВЫЙ ВЫВОД ЧЕРЕЗ ФАЙЛ (обход Python 3.14 бага)
    # =========================================================================
    
    def _run_with_file_logging(self, cmd, description, timeout=7200):
        """Запускает процесс с выводом в файл и потоковым чтением.
        Лог-файл ПЕРЕЗАПИСЫВАЕТСЯ при каждом запуске."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TQDM_DISABLE"] = "1"

        log_path = {
            "training": TRAINING_RUN_LOG_FILE,
            "build": BUILD_RUN_LOG_FILE,
        }.get(description.lower().replace(" ", "_"), os.path.join(LOGS_DIR, f"{description.lower().replace(' ', '_')}.log"))
        
        try:
            # 🆕 ПЕРЕЗАПИСЫВАЕМ лог-файл в начале (режим "w")
            with open(log_path, "w", encoding="utf-8") as log_file:
                self.stop_requested = False
                self.last_run_interrupted = False
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    bufsize=1
                )
                self.active_process = process
                
                # Читаем файл построчно в реальном времени
                last_size = 0
                pending_text = ""
                start_time = time.time()
                last_activity = start_time
                last_heartbeat = start_time
                last_stage = "процесс запущен"
                
                while process.poll() is None:
                    if self.stop_requested:
                        self.last_run_interrupted = True
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        self.log("Процесс остановлен по запросу пользователя.")
                        return False

                    time.sleep(0.5)
                    
                    current_size = os.path.getsize(log_path)
                    if current_size > last_size:
                        with open(log_path, "rb") as f:
                            f.seek(last_size)
                            chunk = f.read()
                        last_size = current_size
                        text = self._decode_log_chunk(chunk)
                        pending_text += text
                        pending_text = pending_text.replace("\r\n", "\n").replace("\r", "\n")
                        lines = pending_text.split("\n")
                        pending_text = lines.pop() if lines else ""

                        emitted = False
                        for line in lines:
                            line = line.strip()
                            if line:
                                if self._should_emit_log_line(line):
                                    self.log(line)
                                    last_stage = self._summarize_stage(line, last_stage)
                                    emitted = True
                        if emitted:
                            last_activity = time.time()
                            last_heartbeat = last_activity
                    elif time.time() - last_heartbeat >= 15:
                        last_heartbeat = time.time()
                    
                    if time.time() - start_time > timeout:
                        process.kill()
                        self.active_process = None
                        self.log(f"Превышено время ожидания ({timeout} сек)")
                        return False
                
                with open(log_path, "rb") as f:
                    f.seek(last_size)
                    chunk = f.read()
                pending_text += self._decode_log_chunk(chunk)
                pending_text = pending_text.replace("\r\n", "\n").replace("\r", "\n")
                for line in pending_text.split("\n"):
                    line = line.strip()
                    if line:
                        if self._should_emit_log_line(line):
                            self.log(line)
                
                return process.returncode == 0
                
        except Exception as e:
            self.log(f"Ошибка: {e}")
            return False
        finally:
            self.active_process = None

    def _latest_checkpoint_dir(self):
        if not os.path.exists(LORA_OUTPUT_DIR):
            return None
        checkpoints = [name for name in os.listdir(LORA_OUTPUT_DIR) if name.startswith("checkpoint-")]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda value: int(value.split("-")[1]))
        return os.path.join(LORA_OUTPUT_DIR, checkpoints[-1])

    def _count_jsonl_rows(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                return sum(1 for line in file_handle if line.strip())
        except OSError:
            return None

    def _get_model_answer(self, question, max_tokens=220):
        local_brain = getattr(self.assistant, "local_brain", None)
        if not local_brain:
            if os.path.exists(GGUF_MODEL_PATH):
                return None, "Собранная модель есть, но сейчас не загружена. Сначала выполните /модель, потом повторите проверку."
            return None, "Сейчас загруженной локальной модели нет. Сначала завершите /собрать, затем выполните /модель."

        try:
            if hasattr(local_brain, "create_chat_completion"):
                result = local_brain.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Отвечай кратко и по-русски."},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                answer = result["choices"][0]["message"]["content"].strip()
            elif hasattr(local_brain, "create_completion"):
                result = local_brain.create_completion(
                    prompt=f"### Вопрос:\n{question}\n\n### Ответ:\n",
                    temperature=0.2,
                    max_tokens=max_tokens,
                    stop=["###"],
                )
                answer = result["choices"][0].get("text", "").strip()
            else:
                result = local_brain(
                    f"### Вопрос:\n{question}\n\n### Ответ:\n",
                    temperature=0.2,
                    max_tokens=max_tokens,
                    stop=["###"],
                )
                answer = result["choices"][0].get("text", "").strip()
        except Exception as e:
            return None, f"Не удалось спросить текущую модель: {e}"

        if not answer:
            return None, "Модель ответила пусто. Попробуйте переформулировать вопрос или увеличить контекст через /ctx."
        return answer, None

    def _ask_current_model(self, question):
        answer, error = self._get_model_answer(question)
        if error:
            return error
        return f"[ПРОВЕРКА МОДЕЛИ]\nВопрос: {question}\nОтвет: {answer}"

    def _run_control_test_suite(self):
        total = len(CONTROL_TEST_QUESTIONS)
        self.log(f"Статус: запущен тест модели, вопросов: {total}.")
        latest_checkpoint = self._latest_checkpoint_dir()
        checkpoint_name = os.path.basename(latest_checkpoint) if latest_checkpoint else "нет"
        self.log(f"Статус: текущий checkpoint для оценки: {checkpoint_name}.")

        results = []
        for index, question in enumerate(CONTROL_TEST_QUESTIONS, start=1):
            if self.stop_requested:
                self.last_run_interrupted = True
                self.log("Тест модели остановлен по запросу пользователя.")
                return
            self.log(f"Статус: тест {index}/{total} — {question}")
            answer, error = self._get_model_answer(question, max_tokens=160)
            final_answer = error or answer or "<пустой ответ>"
            results.append((question, final_answer))
            self.log(f"ТЕСТ {index}/{total} | Вопрос: {question}")
            self.log(f"ТЕСТ {index}/{total} | Ответ: {final_answer}")

        self.log("=" * 68)
        self.log("ИТОГ КОНТРОЛЬНОГО ТЕСТА МОДЕЛИ")
        self.log("=" * 68)
        for index, (question, answer) in enumerate(results, start=1):
            self.log(f"{index}. {question}")
            self.log(f"   {answer}")

    def _decode_log_chunk(self, chunk):
        for encoding in ("utf-8", "cp1251"):
            try:
                return chunk.decode(encoding)
            except UnicodeDecodeError:
                continue
        return chunk.decode("utf-8", errors="replace")

    def _should_emit_log_line(self, line):
        if any(c in line for c in ['█', '▏', '▎', '▍', '▌', '▋', '▊', '▉']) and line.endswith('it/s]'):
            return False
        if re.match(r"^\{'loss':\s*.+\}$", line):
            return False
        return True

    def _summarize_stage(self, line, default_stage):
        important_prefixes = (
            "ШАГ ",
            "Статус:",
            "Загрузка модели",
            "Применение LoRA",
            "Загрузка датасета",
            "Токенизация датасета",
            "Размер обучающего датасета",
            "НАЙДЕН ЧЕКПОИНТ",
            "ЧЕКПОИНТ НЕ НАЙДЕН",
            "НАЧАЛО ОБУЧЕНИЯ",
            "Сохранение модели",
            "ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО",
        )
        if line.startswith(important_prefixes):
            return line
        return default_stage
    
    def skill_training(self, args="", callback_dict=None):
        """Запускает дообучение + автосборку с потоковым выводом."""
        self.log("Запуск дообучения...")
        if not os.path.exists("echo_learn.py"):
            self.log("Файл echo_learn.py не найден.")
            return
        self.log("Обучение запущено (может занять 10-120 минут)...")
        self.log(f"Лог пишется в: {os.path.abspath(self.log_file)}")

        success = self._run_with_file_logging(
            [sys.executable, "echo_learn.py"],
            "training",
            timeout=7200
        )
        
        if success:
            self.log("Обучение завершено!")
            self.log("Автоматически запускаю сборку GGUF...")
            self.skill_build_gguf(args, callback_dict)
        elif self.last_run_interrupted:
            self.log("Обучение прервано пользователем.")
        else:
            self.log("Обучение завершилось с ошибкой")
    
    def skill_build_gguf(self, args="", callback_dict=None):
        """Собирает GGUF с потоковым выводом."""
        self.log("Начало сборки GGUF...")
        if not os.path.exists("build_and_run.py"):
            self.log("Файл build_and_run.py не найден.")
            return
        success = self._run_with_file_logging(
            [sys.executable, "build_and_run.py", "--no-gui"],
            "build",
            timeout=3600
        )
        
        if success:
            self.log("GGUF собрана!")
            self.log("Перезапустите Эхо и в чате напишите: /модель")
        elif self.last_run_interrupted:
            self.log("Сборка прервана пользователем.")
        else:
            self.log("Сборка завершилась с ошибкой")
