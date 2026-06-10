import json
import sys
from collections import Counter
from pathlib import Path

from config import CONFIG
from post_process import finalize, update_main_app
from project_paths import TRAINING_DATA_FILE
from .builder import run_build_pipeline
from .dataset import DataPreparator
from .trainer_runtime import configure_console_utf8, run_training_job


def print_banner():
    mode = "ТЕСТОВЫЙ" if CONFIG["TEST_MODE"] else "ПОЛНОЦЕННЫЙ"
    print(
        f"""
========================================================================
ЭХО: ПОЛНЫЙ ЦИКЛ ДООБУЧЕНИЯ
Режим: {mode} | Примеров: {CONFIG['max_examples']} | Эпох: {CONFIG['epochs']}
Повторный запуск может использовать уже готовый train_data.jsonl.
========================================================================
"""
    )


def run_full_pipeline():
    configure_console_utf8()
    print_banner()
    try:
        dataset_path = prepare_or_reuse_dataset()
        if not dataset_path:
            print("\nОбучение отменено.")
            return False

        if not run_training_job():
            print("\nОбучение не удалось. Сборка пропущена.")
            return False

        finalize()
        update_main_app()
        if not run_build_pipeline(skip_gui=True):
            print("\nОбучение завершилось, но сборка GGUF не удалась.")
            return False

        print("\nПОЛНЫЙ ЦИКЛ УСПЕШНО ЗАВЕРШЁН.")
        return True
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
        return False
    except Exception as exc:
        print(f"\nКритическая ошибка: {exc}")
        return False


def prepare_or_reuse_dataset():
    dataset_path = Path(TRAINING_DATA_FILE)
    if CONFIG.get("reuse_existing_dataset", True) and dataset_path.exists():
        quality_issue = _detect_dataset_quality_issue(dataset_path)
        if not quality_issue:
            print("\nШАГ 2: ИСПОЛЬЗУЮ ГОТОВЫЙ ДАТАСЕТ")
            print("=" * 70)
            print(f"Использую существующий файл: {TRAINING_DATA_FILE}")
            print(f"Примеров в файле: {_count_dataset_rows(dataset_path)}")
            print("Чтобы пересобрать датасет с нуля, установите ECHO_REBUILD_DATASET=1.\n")
            return TRAINING_DATA_FILE

        print("\nШАГ 2: ПЕРЕСБОРКА ДАТАСЕТА")
        print("=" * 70)
        print(f"Найден существующий датасет: {TRAINING_DATA_FILE}")
        print(f"Причина пересборки: {quality_issue}")
        print("Старый датасет не подходит для качественного дообучения.\n")
    return DataPreparator().prepare()


def _count_dataset_rows(path):
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return sum(1 for line in file_handle if line.strip())
    except OSError:
        return 0


def _detect_dataset_quality_issue(path):
    source_counts = Counter()
    total_rows = 0
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                source_counts[str(row.get("source") or "unknown")] += 1
                total_rows += 1
    except (OSError, json.JSONDecodeError):
        return "файл датасета повреждён или не читается"

    if total_rows < CONFIG["min_dataset_size"]:
        return f"слишком мало примеров ({total_rows})"
    if len(source_counts) <= 1:
        return f"в датасете только один источник: {next(iter(source_counts), 'unknown')}"

    dominant_source, dominant_count = source_counts.most_common(1)[0]
    if dominant_count / max(total_rows, 1) >= 0.8:
        share = round((dominant_count / total_rows) * 100, 1)
        return f"источник {dominant_source} занимает {share}% датасета"
    return None


def full_main():
    raise SystemExit(0 if run_full_pipeline() else 1)
