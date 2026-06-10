# post_process.py
import json
import os
from datetime import datetime

from config import CONFIG
from project_paths import MODEL_INFO_FILE

def finalize():
    print("\n🔄 ШАГ 5: ЗАВЕРШЕНИЕ И ИНТЕГРАЦИЯ")
    print("-" * 50)
    os.makedirs(os.path.dirname(MODEL_INFO_FILE), exist_ok=True)
    model_info = {
        "model_path": CONFIG['output_dir'],
        "base_model": CONFIG['base_model'],
        "created": str(datetime.now())
    }
    with open(MODEL_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print("✅ Информация о модели сохранена.")

def update_main_app():
    main_file = "Alter_Echo.py"
    if not os.path.exists(main_file):
        print(f"⚠️ Файл {main_file} не найден, пропускаем обновление.")
        return

    print(
        f"✅ Основной файл {main_file} уже умеет загружать GGUF.\n"
        f"   После сборки модель должна появиться по пути: {CONFIG['gguf_output']}"
    )
