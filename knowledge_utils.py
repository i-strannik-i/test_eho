import json
import re
from pathlib import Path

from project_paths import SUPPORTED_KNOWLEDGE_SUFFIXES


READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251")


def iter_knowledge_files(directory):
    base_dir = Path(directory)
    if not base_dir.exists():
        return []
    return [
        path
        for path in sorted(base_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_SUFFIXES
    ]


def read_knowledge_file(path):
    path = Path(path)
    suffix = path.suffix.lower()

    text = _read_text_with_fallback(path)
    if text is None:
        return ""

    try:
        if suffix in {".txt", ".md", ".markdown", ".csv", ".tsv"}:
            return text
        if suffix == ".json":
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        if suffix == ".jsonl":
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.dumps(json.loads(line), ensure_ascii=False))
                except json.JSONDecodeError:
                    lines.append(line)
            return "\n".join(lines)
    except Exception:
        return text

    return text


def split_large_text(text, chunk_chars, overlap, min_chunk_chars):
    text = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not text:
        return []

    chunks = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            step = max(1, chunk_chars - overlap)
            for start in range(0, len(paragraph), step):
                piece = paragraph[start:start + chunk_chars].strip()
                if len(piece) >= min_chunk_chars:
                    chunks.append(piece)
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = paragraph

    if buffer:
        chunks.append(buffer.strip())

    return [chunk for chunk in chunks if len(chunk) >= min_chunk_chars]


def paragraphs_from_text(text, min_chars=40):
    text = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not text:
        return []
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if len(paragraph.strip()) >= min_chars]


def _read_text_with_fallback(path):
    for encoding in READ_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
