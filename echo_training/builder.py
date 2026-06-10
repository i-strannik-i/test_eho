#!/usr/bin/env python3

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import CONFIG
from project_paths import GGUF_MODEL_PATH, LLAMA_CPP_DIR, LORA_OUTPUT_DIR, MERGED_MODEL_DIR


def configure_console_utf8():
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def run_build_pipeline(skip_gui=True, skip_merge=False, skip_quantize=False):
    configure_console_utf8()

    print("=" * 70)
    print("СБОРКА ДООБУЧЕННОЙ МОДЕЛИ ЭХО")
    print("=" * 70)

    extract_lora_archive()
    if not Path(LORA_OUTPUT_DIR).exists():
        print(f"Папка с LoRA-весами не найдена: {LORA_OUTPUT_DIR}")
        return False

    if not skip_merge:
        if not merge_lora_weights():
            return False
    elif not Path(MERGED_MODEL_DIR).exists():
        print("Пропущено слияние, но объединённая модель отсутствует.")
        return False

    if not convert_to_gguf(skip_quantize=skip_quantize):
        return False

    print("\nШАГ 5: ЗАВЕРШЕНИЕ")
    print(f"GGUF модель: {GGUF_MODEL_PATH}")
    if skip_gui:
        print("Перезапустите Эхо вручную и выполните /модель.")
        return True

    app_file = Path("Alter_Echo.py")
    if app_file.exists():
        return run_command([sys.executable, str(app_file)], "Запуск Alter_Echo.py")

    print("Файл Alter_Echo.py не найден, запуск GUI пропущен.")
    return True


def build_main(argv=None):
    parser = argparse.ArgumentParser(description="Сборка дообученной модели Эхо")
    parser.add_argument("--no-gui", action="store_true", help="Не запускать GUI после сборки")
    parser.add_argument("--skip-merge", action="store_true", help="Пропустить слияние модели")
    parser.add_argument("--skip-quantize", action="store_true", help="Пропустить квантование (оставить F16)")
    args = parser.parse_args(argv)
    success = run_build_pipeline(
        skip_gui=args.no_gui,
        skip_merge=args.skip_merge,
        skip_quantize=args.skip_quantize,
    )
    return 0 if success else 1


def extract_lora_archive(zip_name="echo-lora-weights.zip"):
    zip_path = Path(zip_name)
    lora_dir = Path(LORA_OUTPUT_DIR)
    if not zip_path.exists() or lora_dir.exists():
        return

    print(f"Найден архив весов: {zip_name}")
    print("Распаковка LoRA-весов...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


def merge_lora_weights():
    print("\nШАГ 3: СЛИЯНИЕ БАЗОВОЙ МОДЕЛИ И LORA")
    print("=" * 70)
    merged_dir = Path(MERGED_MODEL_DIR)
    if merged_dir.exists() and any(merged_dir.iterdir()):
        print(f"Объединённая модель уже существует: {MERGED_MODEL_DIR}")
        return True

    lora_source_dir = resolve_lora_source_dir()
    if not lora_source_dir:
        print("Не найдены LoRA-веса для сборки.")
        print(f"Проверена папка: {LORA_OUTPUT_DIR}")
        print("Ожидался файл adapter_config.json в корне models/echo-lora")
        print("или в последнем checkpoint-* после частично завершённого обучения.")
        print("Сначала дождитесь хотя бы одного checkpoint во время /учиться")
        print("или завершите обучение полностью.")
        return False

    missing = []
    for module_name, package_name in {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
    }.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("Для сборки не хватает зависимостей:")
        for package_name in missing:
            print(f"  - {package_name}")
        print(f"\nУстановите их заранее: {sys.executable} -m pip install -r requirements.txt")
        return False

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_gpu = CONFIG.get("profile") == "gpu" and bool(torch.cuda.is_available())
    model_dtype = torch.float16 if use_gpu else torch.float32
    device_map = "auto" if use_gpu else "cpu"

    print("Загрузка базовой модели...")
    print(f"Профиль сборки: {'GPU' if use_gpu else 'CPU'}")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=model_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)

    print("Применение LoRA-весов...")
    print(f"Источник LoRA-весов: {lora_source_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        str(lora_source_dir),
        torch_dtype=model_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print("Слияние завершено.")
    return True


def resolve_lora_source_dir():
    root_dir = Path(LORA_OUTPUT_DIR)
    if has_lora_adapter_files(root_dir):
        return root_dir

    checkpoints = sorted(
        [path for path in root_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[1]),
    ) if root_dir.exists() else []

    for checkpoint_dir in reversed(checkpoints):
        if has_lora_adapter_files(checkpoint_dir):
            return checkpoint_dir
    return None


def has_lora_adapter_files(path):
    if not path or not path.exists() or not path.is_dir():
        return False
    return (path / "adapter_config.json").exists()


def convert_to_gguf(skip_quantize=False):
    print("\nШАГ 4: КОНВЕРТАЦИЯ В GGUF")
    print("=" * 70)

    final_gguf = Path(GGUF_MODEL_PATH)
    if final_gguf.exists():
        print(f"GGUF уже существует: {GGUF_MODEL_PATH}")
        return True

    if not Path(LLAMA_CPP_DIR).exists() and not download_llama_cpp():
        return False

    f16_model_path = Path("models/echo-merged-f16.gguf")
    if not f16_model_path.exists():
        convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            print(f"Не найден конвертер GGUF: {convert_script}")
            return False
        if not run_command(
            [sys.executable, str(convert_script), MERGED_MODEL_DIR, "--outfile", str(f16_model_path), "--outtype", "f16"],
            "Конвертация в F16",
        ):
            return False

    if skip_quantize:
        final_gguf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(f16_model_path), str(final_gguf))
        print(f"F16-модель сохранена как {GGUF_MODEL_PATH}")
        return True

    quantize_exe = download_quantize_tool()
    if not quantize_exe:
        print("Не удалось получить llama-quantize, используем F16 без квантования.")
        final_gguf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(f16_model_path), str(final_gguf))
        return True

    final_gguf.parent.mkdir(parents=True, exist_ok=True)
    if run_command([quantize_exe, str(f16_model_path), str(final_gguf), "q4_K_M"], "Квантование модели"):
        if f16_model_path.exists():
            f16_model_path.unlink()
        return True
    return False


def run_command(cmd, desc="", cwd=None):
    print(f"\n{desc}")
    print(f"   > {' '.join(_quote_cmd_part(part) for part in cmd)}")
    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        print(f"Ошибка при выполнении команды (код {result.returncode}).")
        return False
    return True


def download_llama_cpp():
    url = "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip"
    zip_path = Path("llama_cpp_master.zip")
    extract_dir = Path("llama_cpp_temp")

    print("Скачивание llama.cpp...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_folder = extract_dir / "llama.cpp-master"
        if not extracted_folder.exists():
            print("Не удалось найти распакованную папку llama.cpp-master")
            return False

        target_dir = Path(LLAMA_CPP_DIR)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(extracted_folder), str(target_dir))
        return True
    except Exception as exc:
        print(f"Ошибка скачивания llama.cpp: {exc}")
        return False
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


def download_quantize_tool():
    print("Поиск llama-quantize в релизах GitHub...")
    try:
        url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            release_data = json.loads(response.read().decode("utf-8"))

        asset_url = None
        for asset in release_data.get("assets", []):
            name = asset["name"].lower()
            if "win" in name and "x64" in name and name.endswith(".zip"):
                asset_url = asset["browser_download_url"]
                break

        if not asset_url:
            return None

        zip_path = Path("llama_quantize_win.zip")
        extract_dir = Path("quantize_tool")
        urllib.request.urlretrieve(asset_url, zip_path)

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        for root, _, files in os.walk(extract_dir):
            for filename in files:
                lowered = filename.lower()
                if lowered in {"llama-quantize.exe", "quantize.exe"}:
                    return str(Path(root) / filename)
        return None
    except Exception as exc:
        print(f"Ошибка скачивания квантовщика: {exc}")
        return None
    finally:
        if Path("llama_quantize_win.zip").exists():
            Path("llama_quantize_win.zip").unlink()


def _quote_cmd_part(part):
    return f'"{part}"' if " " in str(part) else str(part)
