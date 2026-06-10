import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import CONFIG
import echo_training.dataset as dataset_module
from echo_training.dataset import DataPreparator


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    temp_root = Path(tempfile.mkdtemp(prefix="echo-dataset-", dir=os.getcwd()))
    db_path = temp_root / "memory.db"
    knowledge_dir = temp_root / "knowledge_input"
    output_path = temp_root / "train_data.jsonl"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE memory (id INTEGER PRIMARY KEY AUTOINCREMENT, user_text TEXT, answer TEXT, weight REAL)")
    cursor.execute("CREATE TABLE logic_laws (premise TEXT, consequence TEXT, solution TEXT, weight REAL)")
    cursor.execute("CREATE TABLE learned_knowledge (content TEXT, knowledge_type TEXT, weight REAL)")
    cursor.execute("INSERT INTO memory (user_text, answer, weight) VALUES (?, ?, ?)", ("привет", "Здравствуйте, чем помочь?", 1.0))
    cursor.execute(
        "INSERT INTO logic_laws (premise, consequence, solution, weight) VALUES (?, ?, ?, ?)",
        ("идет дождь", "будет мокро", "взять зонт", 1.0),
    )
    cursor.execute(
        "INSERT INTO learned_knowledge (content, knowledge_type, weight) VALUES (?, ?, ?)",
        ("Эхо умеет разбирать локальные файлы знаний и сохранять факты.", "fact", 1.0),
    )
    conn.commit()
    conn.close()

    (knowledge_dir / "notes.md").write_text(
        (
            "# Заметки\n\n"
            "Эхо должен приоритизировать локальные знания пользователя перед внешними датасетами. "
            "Если пользователь загрузил файл в knowledge_input, эти знания должны попадать в датасет раньше "
            "массовых внешних примеров, чтобы дообучение не теряло локальный контекст и особенности проекта."
        ),
        encoding="utf-8",
    )
    (knowledge_dir / "rules.json").write_text(
        json.dumps(
            {
                "rule": (
                    "Если файл большой, его нужно делить на чанки и сохранять смысловые блоки. "
                    "Это помогает не терять структуру документа и делает обучение устойчивее."
                )
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    original_config = dict(CONFIG)
    original_output = dataset_module.TRAINING_DATA_FILE
    original_loader = DataPreparator.download_and_parse_hf_datasets

    try:
        CONFIG["db_path"] = str(db_path)
        CONFIG["knowledge_dir"] = str(knowledge_dir)
        CONFIG["max_examples"] = 12
        CONFIG["min_dataset_size"] = 3

        dataset_module.TRAINING_DATA_FILE = str(output_path)
        DataPreparator.download_and_parse_hf_datasets = lambda self: {"hf_alice": [], "hf_yagpt": []}

        generated_path = DataPreparator().prepare()
        require(generated_path == str(output_path), "Unexpected dataset output path")
        require(output_path.exists(), "Dataset file was not created")

        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(len(rows) >= 3, "Dataset is unexpectedly small")
        require(rows[0]["source"] == "dialogues", "Dialogue samples should appear first in raw source order")
        require(any(row["source"] == "knowledge_files" for row in rows), "Knowledge-file samples are missing")
        require(any(row["source"] == "dialogues" for row in rows), "Dialogue samples are missing")
        require(any(row["source"] == "knowledge_base" for row in rows), "Knowledge-base samples are missing")

        parser = DataPreparator()

        alice_pairs = parser._parse_alice_pairs(
            [
                {"session_id": "s1", "role": "assistant", "content": "служебное", "timestamp": "2024-01-01T09:59:00"},
                {"session_id": "s1", "role": "user", "content": "  как дела? ", "timestamp": "2024-01-01T10:00:00"},
                {"session_id": "s1", "role": "alice", "content": " хорошо ", "timestamp": "2024-01-01T10:00:01"},
                {"session_id": "s1", "role": "user", "content": "что нового?", "timestamp": "2024-01-01T10:00:02"},
                {"session_id": "s1", "role": "assistant", "content": "Пока учусь.", "timestamp": "2024-01-01T10:00:03"},
            ]
        )
        require(alice_pairs == [("как дела?", "хорошо"), ("что нового?", "Пока учусь.")], "Alice parser extracted incorrect pairs")

        yagpt_user, yagpt_answer = parser._parse_yagpt_item(
            {"text": "<|speaker1|>Привет<|speaker2|>Здравствуйте!<|speaker1|>Как дела?"}
        )
        require((yagpt_user, yagpt_answer) == ("Привет", "Здравствуйте!"), "Speaker-tag parser extracted incorrect pair")

        msg_user, msg_answer = parser._parse_yagpt_item(
            {
                "messages": [
                    {"role": "system", "content": "ignore"},
                    {"role": "user", "content": "Что ты умеешь?"},
                    {"role": "assistant", "content": "Помогаю с локальным проектом."},
                ]
            }
        )
        require((msg_user, msg_answer) == ("Что ты умеешь?", "Помогаю с локальным проектом."), "Message-list parser extracted incorrect pair")

        CONFIG["max_examples"] = 10
        mixed_loader = lambda self: {
            "hf_alice": [
                {"instruction": "Поставь таймер на 5 минут", "input": "", "output": "Таймер запущен", "weight": 1.0, "source": "hf_alice"},
                {"instruction": "Поставь таймер на 5 минут", "input": "", "output": "Таймер запущен", "weight": 1.0, "source": "hf_alice"},
                {"instruction": "Поставь таймер на 5 минут", "input": "", "output": "Таймер запущен", "weight": 1.0, "source": "hf_alice"},
                {"instruction": "Включи музыку", "input": "", "output": "Музыка включена", "weight": 1.0, "source": "hf_alice"},
                {"instruction": "Какая погода?", "input": "", "output": "Сегодня солнечно", "weight": 1.0, "source": "hf_alice"},
            ],
            "hf_yagpt": [
                {"instruction": "Объясни, зачем нужен лог обучения", "input": "", "output": "Лог обучения показывает стадию, шаги и checkpoint для контроля процесса.", "weight": 1.0, "source": "hf_yagpt"},
                {"instruction": "Как продолжить обучение после остановки?", "input": "", "output": "Нужно перезапустить обучение, и оно продолжится с последнего checkpoint, если он сохранён.", "weight": 1.0, "source": "hf_yagpt"},
            ],
        }
        DataPreparator.download_and_parse_hf_datasets = mixed_loader
        generated_path = DataPreparator().prepare()
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sources = {row["source"] for row in rows}
        require("hf_alice" in sources and "hf_yagpt" in sources, "Mixed dataset lost one of the HF sources")
        timer_rows = [row for row in rows if row["instruction"] == "Поставь таймер на 5 минут"]
        require(len(timer_rows) <= CONFIG["max_short_instruction_repeats"], "Short-command duplicate cap did not work")
        require(any(row["source"] == "knowledge_base" for row in rows), "Local knowledge should remain in mixed dataset")
    finally:
        CONFIG.clear()
        CONFIG.update(original_config)
        dataset_module.TRAINING_DATA_FILE = original_output
        DataPreparator.download_and_parse_hf_datasets = original_loader
        shutil.rmtree(temp_root, ignore_errors=True)

    print("DATASET_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
