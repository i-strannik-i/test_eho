import glob
import os
import sqlite3
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills import SkillManager


class FakeAssistant:
    def __init__(self):
        self.personality = {
            "name": "Эхо",
            "mood": "curious",
            "energy": 1.0,
            "logic": 0.5,
            "creativity": 0.5,
            "stability": 0.5,
            "curiosity": 0.9,
        }
        self.dynamic_topics = {"проект": 1.0}
        self.associative_links = {"проект->эхо": 0.8}
        self.ethics = {
            "enabled": True,
            "law_0": "Не причинять вред человечеству.",
            "law_1": "Не причинять вред людям и животным.",
            "law_3": "Защищать своё существование без вреда.",
        }
        self.cpu_threads = 4
        self.n_ctx = 2048
        self.cognitive_mode = "stable"
        self.embedding_model = None
        self.local_brain = None
        self.db_lock = threading.RLock()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_schema()
        self._seed_data()

    def _create_schema(self):
        self.cursor.execute("CREATE TABLE memory (user_text TEXT, answer TEXT, weight REAL, uses INTEGER)")
        self.cursor.execute("CREATE TABLE logic_laws (premise TEXT, consequence TEXT, solution TEXT, weight REAL)")
        self.cursor.execute("CREATE TABLE learned_knowledge (content TEXT, knowledge_type TEXT, weight REAL, uses INTEGER)")
        self.cursor.execute("CREATE TABLE risk_flags (topic_key TEXT)")
        self.conn.commit()

    def _seed_data(self):
        self.cursor.execute(
            "INSERT INTO memory (user_text, answer, weight, uses) VALUES (?, ?, ?, ?)",
            ("привет", "Здравствуйте", 1.0, 1),
        )
        self.cursor.execute(
            "INSERT INTO memory (user_text, answer, weight, uses) VALUES (?, ?, ?, ?)",
            ("слабая запись", "временный ответ", 0.2, 0),
        )
        self.cursor.execute(
            "INSERT INTO logic_laws (premise, consequence, solution, weight) VALUES (?, ?, ?, ?)",
            ("идет дождь", "будет мокро", "взять зонт", 1.0),
        )
        self.cursor.execute(
            "INSERT INTO learned_knowledge (content, knowledge_type, weight, uses) VALUES (?, ?, ?, ?)",
            ("Эхо умеет работать локально.", "fact", 1.0, 1),
        )
        self.cursor.execute(
            "INSERT INTO learned_knowledge (content, knowledge_type, weight, uses) VALUES (?, ?, ?, ?)",
            ("Слабое знание", "fact", 0.2, 0),
        )
        self.cursor.execute("INSERT INTO risk_flags (topic_key) VALUES (?)", ("опасность",))
        self.conn.commit()

    def save_config(self):
        return None

    def cleanup_memory(self):
        return None

    def process_knowledge_inbox(self):
        return 0


def wait_until_idle(manager, timeout=10):
    started = time.time()
    while manager.running_skill is not None:
        if time.time() - started > timeout:
            raise TimeoutError(f"Skill '{manager.running_skill}' did not finish in time")
        time.sleep(0.05)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    assistant = FakeAssistant()
    manager = SkillManager(assistant)
    cleared = []

    name, args = manager.parse_command("/помощь")
    require((name, args) == ("помощь", ""), "Failed to parse /помощь")
    name, args = manager.parse_command("скилл стат")
    require((name, args) == ("стат", ""), "Failed to parse 'скилл стат'")

    help_out = manager.execute("помощь")
    require("/учиться" in help_out and "/собрать" in help_out and "/проверить" in help_out and "/тест" in help_out, "Help output missing core commands")

    stats_out = manager.execute("стат")
    require("Диалогов:" in stats_out and "Законов логики:" in stats_out, "Stats output missing counters")

    time_out = manager.execute("время")
    require("Сейчас:" in time_out, "Time output missing timestamp")

    laws_out = manager.execute("законы")
    require("Статус:" in laws_out, "Laws output missing status")

    check_out = manager.execute("проверить")
    require("Последний checkpoint:" in check_out and "Как проверить результат:" in check_out, "Learning check skill failed")
    test_out = manager.execute("тест")
    require("локальной модели нет" in test_out.lower() or "сначала завершите /собрать" in test_out.lower(), "Model test skill fallback failed")

    manager.running_skill = "учиться"
    manager._reset_runtime_state("учиться")
    manager.log("Статус: обучение стартовало. План шагов: 15000")
    manager.log("Статус: началась эпоха 1/3.")
    manager.log("Статус: шаг 5/15000, прошло 10 сек.")
    manager.log("Статус: сохранён checkpoint checkpoint-5.")
    snapshot = manager.get_runtime_snapshot()
    require(snapshot["summary"].startswith("Шаг 5 из 15000, эпоха 1/3, checkpoint: checkpoint-5, время: "), "Runtime training summary parsing failed")
    manager.running_skill = None
    manager._reset_runtime_state()

    clear_out = manager.execute("очистить", callback_dict={"clear_chat": lambda: cleared.append(True)})
    require(clear_out == "Чат очищен." and cleared, "Clear chat skill failed")

    cpu_out = manager.execute("cpu", "auto")
    require("Автонастройка CPU" in cpu_out, "CPU auto skill failed")
    cpu_set_out = manager.execute("cpu", "2")
    require("Потоки CPU изменены" in cpu_set_out and assistant.cpu_threads == 2, "CPU set skill failed")

    ctx_out = manager.execute("ctx", "4096")
    require("Контекстное окно изменено" in ctx_out and assistant.n_ctx == 4096, "CTX skill failed")

    restart_out = manager.execute("restart")
    require("Модель не найдена" in restart_out, "Restart fallback failed")

    for path in glob.glob("echo_export_*.json"):
        os.remove(path)

    manager.execute("экспорт")
    wait_until_idle(manager)
    exports = glob.glob("echo_export_*.json")
    require(len(exports) == 1, "Export skill did not create a file")
    os.remove(exports[0])

    manager.execute("сброс")
    wait_until_idle(manager)
    assistant.cursor.execute("SELECT COUNT(*) FROM memory WHERE weight < 0.5 AND uses = 0")
    require(assistant.cursor.fetchone()[0] == 0, "Memory cleanup did not delete weak memory")
    assistant.cursor.execute("SELECT COUNT(*) FROM learned_knowledge WHERE weight < 0.5 AND uses = 0")
    require(assistant.cursor.fetchone()[0] == 0, "Memory cleanup did not delete weak knowledge")

    manager.execute("учить")
    wait_until_idle(manager)
    require(any("knowledge_input пуста" in entry for entry in manager.skill_log), "Ingest skill log missing empty-folder message")

    manager.execute("модель")
    wait_until_idle(manager)
    require(any("Модель не найдена" in entry for entry in manager.skill_log), "Model switch fallback missing")

    original_runner = manager._run_with_file_logging
    manager._run_with_file_logging = lambda *args, **kwargs: False

    manager.execute("собрать")
    wait_until_idle(manager)
    require(any("Сборка завершилась с ошибкой" in entry for entry in manager.skill_log), "Build skill fallback missing")

    manager.execute("учиться")
    wait_until_idle(manager)
    require(any("Обучение завершилось с ошибкой" in entry for entry in manager.skill_log), "Training skill fallback missing")

    manager._run_with_file_logging = original_runner

    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

    interrupter = SkillManager(FakeAssistant())
    interrupter.running_skill = "учиться"
    interrupter.active_process = FakeProcess()
    interrupted, message = interrupter.interrupt_running_skill()
    require(interrupted, "Interrupt request did not report success")
    require(interrupter.stop_requested, "Interrupt request did not set stop_requested")
    require(interrupter.active_process.terminated, "Interrupt request did not terminate active process")
    require("учиться" in message, "Interrupt response did not mention active skill")

    print("SKILL_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
