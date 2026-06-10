# Alter Echo

Локальный русскоязычный ассистент `Эхо` с:
- GUI на `tkinter`
- памятью в `SQLite`
- системой скиллов
- локальным ingestion знаний
- пайплайном подготовки датасета, LoRA-дообучения и сборки `GGUF`

## Установка

Проект больше не ставит зависимости сам во время запуска. Сначала подготовьте окружение:

```powershell
python -m pip install -r requirements.txt
```

## Структура

```text
Alter_Echo.py           # запуск GUI
echo_app/
  core.py               # ядро ассистента, память, знания, поиск, логика
  gui.py                # интерфейс приложения
echo_training/
  dataset.py            # сбор датасета
  trainer_runtime.py    # прямой запуск обучения без генерации run_training.py
  builder.py            # merge + GGUF
  pipeline.py           # полный сценарий обучения
config.py               # единый training config
project_paths.py        # общие пути проекта
knowledge_utils.py      # чтение и разбиение файлов знаний
skills.py               # slash-команды и управление долгими задачами
echo_learn.py           # полный цикл: подготовка + обучение + сборка
build_and_run.py        # отдельная сборка GGUF
logs/                   # runtime-логи приложения, обучения и сборки
knowledge_input/        # локальные файлы знаний
data/processed/         # train_data.jsonl
models/                 # LoRA, merged, GGUF
```

## Что улучшено

- убрана автоустановка зависимостей из runtime
- обучение больше не генерирует `run_training.py`
- подготовка датасета вынесена в отдельный пакет
- сбор датасета идёт напрямую по источникам без балансировки и дедупликации
- GUI и обучение используют общий слой чтения файлов знаний
- сборка модели больше не пишет временный merge-скрипт

## Запуск

### GUI

```powershell
python Alter_Echo.py
```

### Полный пайплайн

```powershell
python echo_learn.py
```

### Только сборка GGUF

```powershell
python build_and_run.py --no-gui
```

## Знания

Положите файлы в `knowledge_input/`, затем:
- нажмите кнопку `Знания` в GUI
- или используйте `/учить`

Поддерживаются:
- `.txt`
- `.md`
- `.json`
- `.jsonl`
- `.csv`
- `.tsv`

Большие файлы режутся на чанки и используются как в GUI-ingestion, так и в обучении.

## Обучение

Источники датасета:
- `ZennyKenny/yandex-alice-sessions-large-syn`
- `under-tree/prepared-yagpt`
- локальная память из базы
- логические законы
- локальные знания из базы
- файлы из `knowledge_input/`
- встроенные безопасные примеры

Подготовка датасета теперь:
- объединяет все источники в сыром виде
- смешивает источники по квотам, чтобы один датасет не забивал всё обучение
- ограничивает короткие повторы и точные дубли
- ограничивает только общий итоговый размер через `max_examples`

Профили обучения:
- `ECHO_PROFILE=auto` — выбрать профиль автоматически
- `ECHO_PROFILE=cpu` — принудительно CPU-профиль
- `ECHO_PROFILE=gpu` — использовать GPU-профиль, если CUDA доступна

CPU-профиль:
- `max_examples = 1000`
- `epochs = 2`
- `max_seq_length = 256`
- `gradient_accumulation_steps = 1`
- `lora_r = 8`
- `save_steps = 20`

GPU-профиль:
- `max_examples = 4000`
- `epochs = 2`
- `batch_size = 2`
- `gradient_accumulation_steps = 2`
- `max_seq_length = 512`
- `lora_r = 16`
- `save_steps = 50`

Тестовый режим:

```powershell
$env:ECHO_TEST_MODE='1'
python echo_learn.py
```

## Проверка

Быстрая проверка того, чему уже успела научиться модель:

```text
/проверить
/проверить что такое Эхо?
/проверить тест
/тест
```

Команда покажет:
- есть ли `checkpoint-*`
- собрана ли `GGUF`-модель
- загружена ли она в Эхо
- и сможет задать контрольный вопрос текущей локальной модели

Режим `/тест` запускает расширенный набор контрольных вопросов и пишет ответы модели одним блоком в лог и окно чата.

Smoke-тест навыков:

```powershell
python tests/test_skills_smoke.py
```

Smoke-тест подготовки датасета:

```powershell
python tests/test_dataset_smoke.py
```

Проверка синтаксиса:

```powershell
python -m py_compile Alter_Echo.py echo_app\core.py echo_app\gui.py echo_training\dataset.py echo_training\trainer_runtime.py echo_training\builder.py echo_training\pipeline.py skills.py tests\test_skills_smoke.py tests\test_dataset_smoke.py
```

## Git

В `.gitignore` уже исключены:
- локальная база памяти
- живой конфиг пользователя
- логи
- веса модели
- датасеты
- временные артефакты

Перед публикацией проверьте:
- что в `knowledge_input/` нет личных файлов
- что в `models/` нет больших бинарников, если не используется Git LFS
- что `assistant_config_v11.json` и `unified_memory_v11.db` не нужны в репозитории
