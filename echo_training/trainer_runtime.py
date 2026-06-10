import inspect
import io
import os
import sys
import time

from config import CONFIG
from project_paths import LORA_OUTPUT_DIR, TRAINING_DATA_FILE

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass


REQUIRED_PACKAGES = {
    "torch": "torch",
    "datasets": "datasets",
    "peft": "peft",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
}


def run_training_job():
    configure_console_utf8()
    missing = missing_packages()
    if missing:
        print("Отсутствуют зависимости для обучения:")
        for package_name in missing:
            print(f"  - {package_name}")
        print(f"\nУстановите их заранее: {sys.executable} -m pip install -r requirements.txt")
        return False

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not os.path.exists(TRAINING_DATA_FILE):
        raise FileNotFoundError(f"Не найден датасет: {TRAINING_DATA_FILE}")

    print("=" * 60)
    print("ОБУЧЕНИЕ ЧЕРЕЗ transformers.Trainer")
    print("=" * 60)

    dtype_value = getattr(torch, CONFIG["torch_dtype"])
    requested_profile = CONFIG.get("requested_profile", CONFIG["profile"])
    active_profile = CONFIG["profile"]
    profile_note = active_profile.upper()
    if requested_profile == "auto":
        profile_note = f"AUTO -> {active_profile.upper()}"
    elif requested_profile != active_profile:
        profile_note = f"{requested_profile.upper()} -> {active_profile.upper()}"

    print(f"Профиль обучения: {profile_note}")
    print(f"Параметры: примеров={CONFIG['max_examples']}, эпох={CONFIG['epochs']}, seq={CONFIG['max_seq_length']}, batch={CONFIG['batch_size']}, accum={CONFIG['gradient_accumulation_steps']}")
    print(f"Загрузка модели ({CONFIG['torch_dtype']} для {active_profile.upper()})...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=dtype_value,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    print("Применение LoRA адаптеров...")
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Загрузка датасета...")
    dataset = load_dataset("json", data_files=TRAINING_DATA_FILE, split="train")
    dataset = dataset.filter(lambda example: bool(example.get("instruction")) and bool(example.get("output")))
    if CONFIG["TEST_MODE"]:
        dataset = dataset.select(range(min(CONFIG["max_examples"], len(dataset))))
    if len(dataset) == 0:
        raise RuntimeError("После фильтрации не осталось ни одного обучающего примера.")

    def format_prompt(example):
        text = f"### Инструкция:\n{example['instruction']}\n\n### Ответ:\n{example['output']}"
        return text[:800] + tokenizer.eos_token

    def tokenize_example(example):
        tokenized = tokenizer(
            format_prompt(example),
            truncation=True,
            max_length=CONFIG["max_seq_length"],
        )
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    print("Токенизация датасета...")
    train_dataset = dataset.map(tokenize_example, remove_columns=dataset.column_names)
    print(f"Размер обучающего датасета: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=CONFIG["logging_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=3,
        gradient_checkpointing=CONFIG["gradient_checkpointing"],
        max_grad_norm=0.3,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=CONFIG["profile"] == "gpu" and CONFIG["torch_dtype"] == "float16",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.add_callback(ConsoleProgressCallback())

    checkpoint_dir = latest_checkpoint_dir(LORA_OUTPUT_DIR)
    if checkpoint_dir:
        print(f"НАЙДЕН ЧЕКПОИНТ! Возобновляем с: {checkpoint_dir}")
    else:
        print("ЧЕКПОИНТ НЕ НАЙДЕН. Обучение начнётся с нуля.")

    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    print("Сохранение модели...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print("=" * 60)
    print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("=" * 60)
    return True


def configure_console_utf8():
    os.environ["TQDM_DISABLE"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
    elif sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
            write_through=True,
        )
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
    elif sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
            write_through=True,
        )


def missing_packages():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    return missing


def latest_checkpoint_dir(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [name for name in os.listdir(output_dir) if name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda value: int(value.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


class ConsoleProgressCallback(TrainerCallback):
    def __init__(self):
        self.started_at = None
        self.last_reported_step = 0
        self.current_step = 0
        self.current_epoch = 0
        self.current_checkpoint = "нет"

    def _status_line(self, step, total_steps):
        elapsed = int(time.time() - self.started_at) if self.started_at else 0
        epoch = self.current_epoch or 1
        return (
            f"Шаг {step} из {total_steps}, "
            f"эпоха {epoch}/{CONFIG['epochs']}, "
            f"checkpoint: {self.current_checkpoint}, "
            f"время: {elapsed} сек"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.time()
        total_steps = state.max_steps or "?"
        print(self._status_line(0, total_steps))
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch or 0) + 1
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.current_step = (state.global_step or 0) + 1
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step or 0
        self.current_step = step
        total_steps = state.max_steps or "?"
        if step <= 3 or (step % 5 == 0 and step != self.last_reported_step):
            print(self._status_line(step, total_steps))
            self.last_reported_step = step
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']}")
        if parts:
            print(f"Метрики: шаг {state.global_step}/{state.max_steps or '?'} | " + " | ".join(parts))
        return control

    def on_save(self, args, state, control, **kwargs):
        self.current_checkpoint = f"checkpoint-{state.global_step}"
        print(self._status_line(state.global_step or 0, state.max_steps or "?"))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = int(time.time() - self.started_at) if self.started_at else 0
        print(f"Обучение завершено. Время: {elapsed} сек.")
        return control
