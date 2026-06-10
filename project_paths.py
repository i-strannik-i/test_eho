from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

DATABASE_FILE = "unified_memory_v11.db"
ASSISTANT_CONFIG_FILE = "assistant_config_v11.json"
KNOWLEDGE_INPUT_DIR = "knowledge_input"
LOGS_DIR = "logs"

LOG_FILE = f"{LOGS_DIR}/echo_log.txt"
SKILL_LOG_FILE = f"{LOGS_DIR}/echo_training.log"
TRAINING_RUN_LOG_FILE = f"{LOGS_DIR}/training.log"
BUILD_RUN_LOG_FILE = f"{LOGS_DIR}/build.log"

TRAINING_DATA_FILE = "data/processed/train_data.jsonl"
MODEL_INFO_FILE = "models/model_info.json"
LORA_OUTPUT_DIR = "models/echo-lora"
MERGED_MODEL_DIR = "models/echo-merged"
GGUF_MODEL_PATH = "models/qwen-1_5b.gguf"

LLAMA_CPP_DIR = "llama.cpp"

SUPPORTED_KNOWLEDGE_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
}


STARTUP_LOG_FILES = (
    LOG_FILE,
    SKILL_LOG_FILE,
    TRAINING_RUN_LOG_FILE,
    BUILD_RUN_LOG_FILE,
)
