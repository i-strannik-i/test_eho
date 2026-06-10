import os

from project_paths import DATABASE_FILE, GGUF_MODEL_PATH, KNOWLEDGE_INPUT_DIR, LORA_OUTPUT_DIR


TEST_MODE = os.environ.get("ECHO_TEST_MODE", "0") == "1"
REBUILD_DATASET = os.environ.get("ECHO_REBUILD_DATASET", "0") == "1"
REQUESTED_PROFILE = os.environ.get("ECHO_PROFILE", "auto").strip().lower()
if REQUESTED_PROFILE not in {"auto", "cpu", "gpu"}:
    REQUESTED_PROFILE = "auto"


def detect_training_profile(requested_profile):
    if requested_profile == "cpu":
        return "cpu"
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if requested_profile == "gpu":
        return "gpu" if has_cuda else "cpu"
    return "gpu" if has_cuda else "cpu"


PROFILE_SETTINGS = {
    "cpu": {
        "max_examples": 1000,
        "epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "max_seq_length": 256,
        "lora_r": 8,
        "lora_alpha": 16,
        "save_steps": 20,
        "logging_steps": 20,
        "gradient_checkpointing": False,
        "torch_dtype": "float32",
    },
    "gpu": {
        "max_examples": 4000,
        "epochs": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "max_seq_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "save_steps": 50,
        "logging_steps": 10,
        "gradient_checkpointing": True,
        "torch_dtype": "float16",
    },
}
ACTIVE_PROFILE = detect_training_profile(REQUESTED_PROFILE)
ACTIVE_PROFILE_SETTINGS = dict(PROFILE_SETTINGS[ACTIVE_PROFILE])
if TEST_MODE:
    ACTIVE_PROFILE_SETTINGS.update(
        {
            "max_examples": 500,
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 256,
            "save_steps": 10,
            "logging_steps": 10,
        }
    )


CONFIG = {
    "TEST_MODE": TEST_MODE,
    "requested_profile": REQUESTED_PROFILE,
    "profile": ACTIVE_PROFILE,
    "db_path": DATABASE_FILE,
    "knowledge_dir": KNOWLEDGE_INPUT_DIR,
    "base_model": "Qwen/Qwen2-1.5B-Instruct",
    "output_dir": LORA_OUTPUT_DIR,
    "gguf_output": GGUF_MODEL_PATH,
    "reuse_existing_dataset": not REBUILD_DATASET,
    "min_dataset_size": 50,
    "file_chunk_chars": 1200,
    "file_chunk_overlap": 180,
    "min_file_chunk_chars": 220,
    "source_priority": [
        "dialogues",
        "knowledge_base",
        "knowledge_files",
        "logic_laws",
        "ethical",
        "hf_yagpt",
        "hf_alice",
    ],
    "source_ratios": {
        "dialogues": 0.12,
        "knowledge_base": 0.12,
        "knowledge_files": 0.16,
        "logic_laws": 0.06,
        "ethical": 0.01,
        "hf_yagpt": 0.33,
        "hf_alice": 0.20,
    },
    "short_instruction_chars": 40,
    "max_short_instruction_repeats": 2,
}
CONFIG.update(ACTIVE_PROFILE_SETTINGS)
