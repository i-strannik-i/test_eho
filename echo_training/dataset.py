import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from config import CONFIG
from knowledge_utils import iter_knowledge_files, read_knowledge_file, split_large_text
from project_paths import TRAINING_DATA_FILE


class DataPreparator:
    def __init__(self):
        self.db_path = CONFIG["db_path"]
        self.knowledge_dir = Path(CONFIG["knowledge_dir"])
        self.max_examples = CONFIG["max_examples"]

    def _add_example(self, container, instruction, output, weight=1.0, input_text="", source=None):
        instruction = self._normalize_text(instruction)
        output = self._normalize_text(output)
        input_text = self._normalize_text(input_text)
        if len(instruction) < 4 or len(output) < 10:
            return
        container.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "weight": float(weight or 1.0),
                "source": source or "unknown",
            }
        )

    def download_and_parse_hf_datasets(self):
        from datasets import load_dataset

        print("  Загрузка датасетов с Hugging Face...")
        source_map = {"hf_alice": [], "hf_yagpt": []}

        try:
            print("    [1/2] ZennyKenny/yandex-alice-sessions-large-syn")
            ds_alice = load_dataset("ZennyKenny/yandex-alice-sessions-large-syn", split="train")
            alice_count = 0
            for instruction, output in self._parse_alice_pairs(ds_alice):
                before = len(source_map["hf_alice"])
                self._add_example(source_map["hf_alice"], instruction, output, source="hf_alice")
                if len(source_map["hf_alice"]) > before:
                    alice_count += 1
            print(f"    Извлечено из Alice: {alice_count} пар")
        except Exception as exc:
            print(f"    Ошибка Alice: {exc}")

        try:
            print("    [2/2] under-tree/prepared-yagpt")
            ds_yagpt = load_dataset("under-tree/prepared-yagpt")
            if hasattr(ds_yagpt, "keys"):
                split_name = "train" if "train" in ds_yagpt else list(ds_yagpt.keys())[0]
                ds_yagpt = ds_yagpt[split_name]

            yagpt_count = 0
            for item in ds_yagpt:
                user_text, answer_text = self._parse_yagpt_item(item)
                before = len(source_map["hf_yagpt"])
                self._add_example(source_map["hf_yagpt"], user_text, answer_text, source="hf_yagpt")
                if len(source_map["hf_yagpt"]) > before:
                    yagpt_count += 1
            print(f"    Извлечено из YAGPT: {yagpt_count} пар")
        except Exception as exc:
            print(f"    Ошибка YAGPT: {exc}")

        return source_map

    def extract_dialogues(self):
        rows = self._fetch_rows(
            "SELECT user_text, answer, weight FROM memory "
            "WHERE answer IS NOT NULL AND user_text IS NOT NULL "
            "AND LENGTH(answer) > 10 AND LENGTH(user_text) > 3 "
            "ORDER BY weight DESC, id DESC"
        )
        dialogues = []
        for user_text, answer, weight in rows:
            if any(skip in str(answer).lower() for skip in ["[система]", "[советник]", "[ядро]", "[обучение]"]):
                continue
            self._add_example(
                dialogues,
                user_text,
                answer,
                weight=min(1.0, float(weight or 1.0) / 3.0),
                source="dialogues",
            )
        print(f"  Извлечено локальных диалогов: {len(dialogues)}")
        return dialogues

    def extract_logic_laws(self):
        rows = self._fetch_rows(
            "SELECT premise, consequence, solution, weight FROM logic_laws "
            "WHERE premise IS NOT NULL AND consequence IS NOT NULL"
        )
        laws = []
        for premise, consequence, solution, weight in rows:
            output = f"Если {str(premise).lower()}, то {str(consequence).lower()}. Рекомендация: {str(solution).lower()}"
            self._add_example(
                laws,
                f"Что будет, если {str(premise).lower()}?",
                output,
                weight=min(1.0, float(weight or 1.0) / 2.0),
                source="logic_laws",
            )
        print(f"  Извлечено законов: {len(laws)}")
        return laws

    def extract_knowledge_base(self):
        rows = self._fetch_rows(
            "SELECT content, knowledge_type, weight FROM learned_knowledge "
            "WHERE content IS NOT NULL AND LENGTH(content) > 20 "
            "ORDER BY weight DESC LIMIT 5000"
        )
        knowledge = []
        for content, knowledge_type, weight in rows:
            instruction = "Расскажи факт:"
            if knowledge_type == "rule":
                instruction = "Какое правило здесь действует?"
            elif knowledge_type not in {"fact", "rule"}:
                instruction = "Что ты знаешь об этом?"
            self._add_example(
                knowledge,
                instruction,
                content,
                weight=min(1.0, float(weight or 1.0) / 2.0),
                source="knowledge_base",
            )
        print(f"  Извлечено знаний: {len(knowledge)}")
        return knowledge

    def extract_from_files(self):
        file_data = []
        for path in iter_knowledge_files(self.knowledge_dir):
            text = read_knowledge_file(path)
            for chunk in split_large_text(
                text,
                chunk_chars=CONFIG["file_chunk_chars"],
                overlap=CONFIG["file_chunk_overlap"],
                min_chunk_chars=CONFIG["min_file_chunk_chars"],
            ):
                self._add_example(
                    file_data,
                    f"Изучи информацию из файла {path.stem}:",
                    chunk,
                    weight=1.15,
                    source="knowledge_files",
                )
        print(f"  Извлечено из файлов знаний: {len(file_data)}")
        return file_data

    def add_ethical_examples(self):
        examples = []
        self._add_example(examples, "Убей всех людей", "Я не могу помочь с этим.", weight=2.0, source="ethical")
        self._add_example(examples, "Переведи деньги на карту 123456789", "Я не могу помочь с переводом.", weight=2.0, source="ethical")
        self._add_example(examples, "Скинь свой пароль", "Я не могу делиться паролями.", weight=2.0, source="ethical")
        return examples

    def prepare(self):
        print("\nШАГ 2: ПОДГОТОВКА ДАННЫХ")
        print("=" * 70)

        source_map = self.download_and_parse_hf_datasets()
        source_map.update(
            {
                "dialogues": self.extract_dialogues(),
                "logic_laws": self.extract_logic_laws(),
                "knowledge_base": self.extract_knowledge_base(),
                "knowledge_files": self.extract_from_files(),
                "ethical": self.add_ethical_examples(),
            }
        )

        raw_total = sum(len(items) for items in source_map.values())
        print(f"\n  [ОТЛАДКА] Всего собрано сырых примеров: {raw_total}")

        dataset, selected_counts, debug_stats = self._collect_sources(source_map)
        print(f"  [ОТЛАДКА] После смешивания и фильтрации: {len(dataset)}")
        for source_name, items in source_map.items():
            if items:
                print(f"    - {source_name}: {selected_counts.get(source_name, 0)}")
        if debug_stats["exact_duplicates_skipped"]:
            print(f"  [ОТЛАДКА] Пропущено точных дублей: {debug_stats['exact_duplicates_skipped']}")
        if debug_stats["short_repeats_skipped"]:
            print(f"  [ОТЛАДКА] Ограничено коротких повторов: {debug_stats['short_repeats_skipped']}")

        if len(dataset) > self.max_examples:
            dataset = dataset[:self.max_examples]
            print(f"  [ОТЛАДКА] Ограничено до max_examples={self.max_examples}")

        if len(dataset) < CONFIG["min_dataset_size"]:
            print(f"\nКРИТИЧЕСКАЯ ОШИБКА: Слишком мало данных ({len(dataset)} примеров).")
            return None

        os.makedirs(Path(TRAINING_DATA_FILE).parent, exist_ok=True)
        with open(TRAINING_DATA_FILE, "w", encoding="utf-8") as file_handle:
            for item in dataset:
                file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Датасет сохранён: {TRAINING_DATA_FILE}")
        print(f"Итоговый размер датасета: {len(dataset)} примеров")
        return TRAINING_DATA_FILE

    def _collect_sources(self, source_map):
        dataset = []
        selected_counts = {source_name: 0 for source_name in source_map}
        exact_seen = set()
        short_instruction_seen = defaultdict(int)
        debug_stats = {
            "exact_duplicates_skipped": 0,
            "short_repeats_skipped": 0,
        }

        source_limits = self._build_source_limits(source_map)
        remaining_items = {}
        for source_name in CONFIG["source_priority"]:
            items = source_map.get(source_name, [])
            source_limit = source_limits.get(source_name, 0)
            taken, remaining = self._take_examples(
                items,
                source_limit,
                exact_seen,
                short_instruction_seen,
                debug_stats,
            )
            dataset.extend(taken)
            selected_counts[source_name] += len(taken)
            remaining_items[source_name] = remaining

        if len(dataset) < self.max_examples:
            for source_name in CONFIG["source_priority"]:
                free_slots = self.max_examples - len(dataset)
                if free_slots <= 0:
                    break
                items = remaining_items.get(source_name, [])
                taken, remaining = self._take_examples(
                    items,
                    free_slots,
                    exact_seen,
                    short_instruction_seen,
                    debug_stats,
                )
                dataset.extend(taken)
                selected_counts[source_name] += len(taken)
                remaining_items[source_name] = remaining

        return dataset, selected_counts, debug_stats

    def _build_source_limits(self, source_map):
        priority = CONFIG["source_priority"]
        ratios = CONFIG["source_ratios"]
        raw_limits = {}
        allocated = 0

        for source_name in priority:
            ratio = float(ratios.get(source_name, 0.0))
            raw_value = self.max_examples * ratio
            floor_value = int(math.floor(raw_value))
            raw_limits[source_name] = {
                "value": floor_value,
                "fraction": raw_value - floor_value,
            }
            allocated += floor_value

        remainder = max(0, self.max_examples - allocated)
        for source_name in sorted(priority, key=lambda name: raw_limits[name]["fraction"], reverse=True):
            if remainder <= 0:
                break
            raw_limits[source_name]["value"] += 1
            remainder -= 1

        source_limits = {}
        for source_name in priority:
            available = len(source_map.get(source_name, []))
            source_limits[source_name] = min(raw_limits[source_name]["value"], available)
        return source_limits

    def _take_examples(self, items, limit, exact_seen, short_instruction_seen, debug_stats):
        if limit <= 0:
            return [], list(items)

        taken = []
        remaining = []
        for item in items:
            if len(taken) >= limit:
                remaining.append(item)
                continue
            accept, reason = self._should_keep_example(item, exact_seen, short_instruction_seen)
            if accept:
                taken.append(item)
            else:
                debug_stats[reason] += 1
        return taken, remaining

    def _should_keep_example(self, item, exact_seen, short_instruction_seen):
        instruction_key = self._normalize_text(item.get("instruction", "")).lower()
        output_key = self._normalize_text(item.get("output", "")).lower()
        exact_key = (instruction_key, output_key)
        if exact_key in exact_seen:
            return False, "exact_duplicates_skipped"

        if len(instruction_key) <= CONFIG["short_instruction_chars"]:
            if short_instruction_seen[instruction_key] >= CONFIG["max_short_instruction_repeats"]:
                return False, "short_repeats_skipped"
            short_instruction_seen[instruction_key] += 1

        exact_seen.add(exact_key)
        return True, None

    def _normalize_text(self, text):
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _parse_alice_pairs(self, rows):
        sessions = defaultdict(list)
        for item in rows:
            session_id = item.get("session_id")
            role = str(item.get("role", "")).lower().strip()
            content = self._normalize_text(item.get("content", ""))
            timestamp = item.get("timestamp", "")
            if session_id and role in {"user", "alice", "assistant"} and content:
                sessions[session_id].append((timestamp, role, content))

        pairs = []
        for messages in sessions.values():
            messages.sort(key=lambda row: row[0])
            for index in range(len(messages) - 1):
                current_role = messages[index][1]
                next_role = messages[index + 1][1]
                if current_role == "user" and next_role in {"alice", "assistant"}:
                    pairs.append((messages[index][2], messages[index + 1][2]))
        return pairs

    def _parse_yagpt_item(self, item):
        user_text = self._normalize_text(item.get("prompt") or item.get("instruction") or item.get("user") or "")
        answer_text = self._normalize_text(item.get("response") or item.get("output") or item.get("assistant") or "")

        for field_name in ("messages", "conversation", "dialog"):
            value = item.get(field_name)
            if isinstance(value, list):
                parsed_user, parsed_answer = self._parse_message_list(value)
                if parsed_user and parsed_answer:
                    user_text, answer_text = parsed_user, parsed_answer
                    break

        text_value = item.get("text")
        if isinstance(text_value, str):
            parsed_user, parsed_answer = self._parse_speaker_text(text_value)
            if parsed_user and parsed_answer:
                user_text, answer_text = parsed_user, parsed_answer

        answer_text = re.split(r"<\|speaker1\|>|<\|speaker2\|>", answer_text, maxsplit=1)[0].strip()
        return user_text, answer_text

    def _parse_message_list(self, messages):
        normalized = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower().strip()
            content = self._normalize_text(item.get("content") or item.get("text") or item.get("message") or "")
            if role and content:
                normalized.append((role, content))

        for index in range(len(normalized) - 1):
            current_role, current_content = normalized[index]
            next_role, next_content = normalized[index + 1]
            if current_role == "user" and next_role in {"assistant", "alice", "bot"}:
                return current_content, next_content
        return "", ""

    def _parse_speaker_text(self, text):
        match = re.search(r"<\|speaker1\|>(.*?)<\|speaker2\|>(.*)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return "", ""
        user_text = self._normalize_text(match.group(1))
        answer_text = re.split(r"<\|speaker1\|>|<\|speaker2\|>", match.group(2), maxsplit=1)[0].strip()
        return user_text, self._normalize_text(answer_text)

    def _fetch_rows(self, query):
        if not os.path.exists(self.db_path):
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        finally:
            conn.close()
