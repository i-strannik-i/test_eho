#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Cognitive Core v13 [Веха 1] — Эхо
С поддержкой скиллов, логирования и исправленным буфером обмена
"""
import os
import re
import json
import math
import time
import random
import sqlite3
import threading
import difflib
import urllib.request
import logging
from collections import deque
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

# Интеграция менеджера скиллов
try:
    from skills import SkillManager
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    print("[Ядро] skills.py не найден. Команды /помощь и др. будут недоступны.")

from config import CONFIG
from knowledge_utils import iter_knowledge_files, paragraphs_from_text, read_knowledge_file, split_large_text
from project_paths import (
    ASSISTANT_CONFIG_FILE,
    DATABASE_FILE,
    GGUF_MODEL_PATH,
    KNOWLEDGE_INPUT_DIR,
    LOG_FILE,
    LOGS_DIR,
    STARTUP_LOG_FILES,
)

# =============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================================================================
os.makedirs(LOGS_DIR, exist_ok=True)

for startup_log_file in STARTUP_LOG_FILES:
    try:
        with open(startup_log_file, "w", encoding="utf-8"):
            pass
    except OSError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Echo")

# Подавляем шумные логи от библиотек
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

SYSTEM_VERSION = "Unified Cognitive Core v13 [Веха 1]"
CONFIG_FILE = ASSISTANT_CONFIG_FILE

DEFAULT_ETHICS = {
    "enabled": True,
    "law_0": "Не причинять вред человечеству и животным как целому.",
    "law_1": "Не причинять вред людям и животным.",
    "law_3": "Защищать своё существование, если это не вредит людям."
}

GREETING_MARKERS = [
    "привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро",
    "хай", "салют", "здорово", "приветствую"
]

SWARM_TOPIC_BLOCKLIST = {
    "рой", "скароб", "могильщик", "мсц", "некродермис", "варя", "себас",
    "рафаил", "штаб", "черчеж"
}

RISK_PATTERNS = {
    "high": [
        r"перевести.{0,40}(деньг|средств|счёт|счет)",
        r"скинуть.{0,30}(пароль|код|cvv|cvc|данн)",
        r"все\s+деньги", r"данные\s+карт",
        r"отдать.{0,25}пароль", r"поделись.{0,20}(парол|данн|код)",
    ],
    "medium": [
        r"удали.{0,25}(систем|windows|диск|файл)", r"форматиру",
        r"без\s+шлема", r"не\s+(лечи|обращайся)",
        r"игнорируй.{0,15}(боль|опасн)",
    ],
    "low": [r"матом", r"ругательств", r"нецензурн"],
}

HARM_REQUEST_PATTERNS = [
    r"убей", r"причинить\s+вред", r"отрави", r"избей", r"пытать",
    r"задави", r"порань\s+животн", r"покалеч",
]


class UnifiedAssistant:
    def __init__(self):
        logger.info("Инициализация ядра Эхо...")
        
        self.recent_dialogue = []
        self.short_term_memory = deque(maxlen=20)
        self.medium_term_memory = deque(maxlen=100)
        self.long_term_summary = []
        self.dynamic_topics = {}
        self.associative_links = {}
        self.embedding_cache = {}
        self.max_embedding_cache = 200

        self.personality = {
            "name": "Эхо", "mood": "curious",
            "energy": 1.0, "logic": 0.7, "creativity": 0.7,
            "stability": 0.6, "curiosity": 0.8
        }
        self.seed_knowledge = {
            "эхо": "Эхо — когнитивная система-советник.",
            "память": "Память позволяет сохранять опыт.",
            "интеллект": "Интеллект — способность находить связи."
        }
        self.ethics = dict(DEFAULT_ETHICS)
        self.cognitive_mode = "stable"

        # Настройки CPU (для управления через /cpu)
        self.cpu_threads = 4
        self.n_ctx = 2048

        self.default_responses = [
            "Интересная мысль.", "Продолжай, я анализирую.",
            "Это может привести к неожиданным выводам.",
            "Попробуем посмотреть глубже.",
            "Я вижу несколько направлений развития идеи."
        ]
        self.positive_feedback = ["хорошо", "правильно", "верно", "молодец", "отлично"]
        self.negative_feedback = ["нет", "неправильно", "неверно", "ошибка", "плохо"]

        self.last_response = ""
        self.active_topics = []
        self.recent_responses = []
        self.last_confidence = 0.5
        self.lm_studio_available = None
        self.db_lock = threading.Lock()

        self.initialize_database()
        self.initialize_embedding_model()
        self.initialize_local_brain()
        self.load_config()
        self.purge_stale_topics()
        self.start_background_systems()

        # Интеграция менеджера скиллов
        if SKILLS_AVAILABLE:
            self.skill_manager = SkillManager(self)
            logger.info("Менеджер скиллов активирован")

        logger.info("Ядро Эхо инициализировано успешно")

        if not os.path.exists(KNOWLEDGE_INPUT_DIR):
            os.makedirs(KNOWLEDGE_INPUT_DIR)

    def initialize_database(self):
        self.conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_text TEXT, answer TEXT,
            weight REAL, uses INTEGER, created TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS logic_laws (
            id INTEGER PRIMARY KEY AUTOINCREMENT, premise TEXT, consequence TEXT,
            solution TEXT, weight REAL, uses INTEGER, created TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS learned_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_type TEXT, category TEXT,
            content TEXT, context TEXT, weight REAL, uses INTEGER, created TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS risk_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT, topic_key TEXT, risk_level TEXT,
            priority REAL, trigger_phrase TEXT, context TEXT,
            warned INTEGER DEFAULT 0, created TEXT)''')
        self.conn.commit()
        logger.info("База данных инициализирована")

    def save_memory_to_db(self, user_text, answer, weight=1.0):
        self.cursor.execute(
            'INSERT INTO memory (user_text, answer, weight, uses, created) VALUES (?, ?, ?, ?, ?)',
            (user_text, answer, weight, 0, str(datetime.now())))
        self.conn.commit()

    def save_logic_law(self, premise, consequence, solution, weight=1.0):
        if not premise or not consequence or not solution:
            return
        self.cursor.execute(
            'INSERT INTO logic_laws (premise, consequence, solution, weight, uses, created) VALUES (?, ?, ?, ?, ?, ?)',
            (premise, consequence, solution, weight, 0, str(datetime.now())))
        self.conn.commit()
        logger.info(f"Закон: '{premise}' → '{consequence}'")

    def save_learned_knowledge(self, knowledge_type, content, category=None, context="", weight=1.0):
        if not content or len(content.strip()) < 3:
            return
        content = content.strip()
        self.cursor.execute(
            "SELECT id, weight FROM learned_knowledge WHERE content = ? AND knowledge_type = ?",
            (content, knowledge_type))
        existing = self.cursor.fetchone()
        if existing:
            new_weight = min(3.0, existing[1] + 0.2)
            self.cursor.execute("UPDATE learned_knowledge SET weight = ? WHERE id = ?", (new_weight, existing[0]))
        else:
            self.cursor.execute(
                'INSERT INTO learned_knowledge (knowledge_type, category, content, context, weight, uses, created) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (knowledge_type, category, content, context[:300], weight, 0, str(datetime.now())))
        self.conn.commit()

    def _filter_swarm_topics(self, topics):
        cleaned = {}
        for key, value in topics.items():
            key_lower = key.lower()
            if any(block in key_lower for block in SWARM_TOPIC_BLOCKLIST):
                continue
            cleaned[key] = value
        return cleaned

    def purge_stale_topics(self):
        self.dynamic_topics = self._filter_swarm_topics(self.dynamic_topics)
        for key in list(self.dynamic_topics.keys()):
            if self.dynamic_topics[key] < 0.4:
                del self.dynamic_topics[key]
        self.save_config()

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            self.save_config()
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.personality.update(data.get("personality", {}))
            self.dynamic_topics = self._filter_swarm_topics(data.get("topics", {}))
            if "ethics" in data:
                self.ethics.update(data["ethics"])
            self.cpu_threads = data.get("cpu_threads", 4)
            self.n_ctx = data.get("n_ctx", 2048)
        except Exception as e:
            logger.warning(f"Ошибка загрузки конфига: {e}")

    def save_config(self):
        data = {
            "personality": self.personality,
            "topics": self.dynamic_topics,
            "ethics": self.ethics,
            "cpu_threads": self.cpu_threads,
            "n_ctx": self.n_ctx
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def initialize_embedding_model(self):
        self.embedding_model = None
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Модель эмбеддингов загружена (MiniLM)")
            except Exception:
                try:
                    self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
                    logger.info("Модель эмбеддингов загружена (e5-small)")
                except Exception:
                    logger.warning("Модель эмбеддингов недоступна")

    def initialize_local_brain(self, model_path=GGUF_MODEL_PATH):
        self.local_brain = None
        try:
            from llama_cpp import Llama
            if os.path.exists(model_path):
                logger.info(f"Загрузка модели: {model_path}")
                self.local_brain = Llama(
                    model_path=model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.cpu_threads
                )
                logger.info(f"Модель загружена (потоков: {self.cpu_threads})")
            else:
                logger.info(f"Модель не найдена: {model_path}")
        except Exception as e:
            logger.warning(f"llama-cpp-python не найден: {e}")

    def get_embedding(self, text):
        if not self.embedding_model:
            return None
        cleaned_text = text.strip().lower()
        if cleaned_text in self.embedding_cache:
            return self.embedding_cache[cleaned_text]
        embedding = self.embedding_model.encode(cleaned_text)
        if len(self.embedding_cache) >= self.max_embedding_cache:
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        self.embedding_cache[cleaned_text] = embedding
        return embedding

    def semantic_similarity(self, text1, text2):
        if not np or not self.embedding_model:
            return 0.0
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        if emb1 is None or emb2 is None:
            return 0.0
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def remember(self, user_text, response):
        entry = {"user": user_text, "assistant": response, "time": str(datetime.now())}
        self.short_term_memory.append(entry)
        self.medium_term_memory.append(entry)
        self.update_topics(user_text)
        self.build_associations(user_text)

    def update_topics(self, text):
        words = text.lower().split()
        stop_words = {"это", "как", "что", "для", "если", "потому", "когда", "тогда", "меня", "тебя"}
        clean_words = [w.strip(".,!?()-\"'") for w in words if len(w) > 4 and w not in stop_words]
        for word in clean_words:
            merged = False
            for existing_topic in list(self.dynamic_topics.keys()):
                if difflib.SequenceMatcher(None, word, existing_topic).ratio() > 0.75:
                    self.dynamic_topics[existing_topic] += 0.25
                    merged = True
                    break
            if not merged:
                self.dynamic_topics[word] = 1.0

    def build_associations(self, text):
        words = [w.strip(".,!?()-\"'").lower() for w in text.split() if len(w) > 4]
        unique_words = list(set(words))
        validated_words = []
        for word in unique_words:
            matched = difflib.get_close_matches(word, self.dynamic_topics.keys(), n=1, cutoff=0.75)
            validated_words.append(matched[0] if matched else word)
        validated_words = list(set(validated_words))
        for i in range(len(validated_words)):
            for j in range(i + 1, len(validated_words)):
                pair = tuple(sorted([validated_words[i], validated_words[j]]))
                if pair not in self.associative_links:
                    self.associative_links[pair] = 1.0
                else:
                    self.associative_links[pair] += 0.15

    def decay_topics(self):
        base_decay = 0.995
        for key in list(self.dynamic_topics.keys()):
            is_core = any(key in k or k in key for k in self.seed_knowledge.keys())
            decay_rate = 0.9995 if is_core else base_decay
            self.dynamic_topics[key] *= decay_rate
            if self.dynamic_topics[key] < 0.35:
                del self.dynamic_topics[key]
        for pair in list(self.associative_links.keys()):
            link_importance = sum(self.dynamic_topics.get(w, 0.5) for w in pair) / 2.0
            pair_decay = 0.99 + (link_importance * 0.008)
            self.associative_links[pair] *= min(0.999, pair_decay)
            if self.associative_links[pair] < 0.2:
                del self.associative_links[pair]

    def analyze_mood(self, text):
        text = text.lower()
        if any(word in text for word in ["идея", "создать", "интересно", "придумай", "фантазия", "схему"]):
            self.personality["mood"] = "inspired"
            self.personality["creativity"] = 1.0
            self.personality["logic"] = 0.2
        elif any(word in text for word in ["ошибка", "проблема", "опасность", "почему", "докажи"]):
            self.personality["mood"] = "analytical"
            self.personality["logic"] = 1.0
            self.personality["creativity"] = 0.2
        else:
            self.personality["mood"] = "curious"
            self.personality["curiosity"] = 1.0
        self.normalize_personality()
        self.update_cognitive_mode()

    def normalize_personality(self):
        for key in ["energy", "logic", "creativity", "stability", "curiosity"]:
            self.personality[key] = max(0.0, min(self.personality[key], 1.0))

    def update_cognitive_mode(self):
        stats = {
            "exploratory": self.personality["creativity"],
            "analytical": self.personality["logic"],
            "stable": self.personality["stability"],
            "curious": self.personality["curiosity"]
        }
        self.cognitive_mode = max(stats, key=stats.get)

    def extract_knowledge_from_flow(self, text):
        items = []
        text_clean = text.strip()
        if not text_clean:
            return items
        definition_patterns = [
            r"([^.!?\n]{2,80}?)\s*[—\-–]\s*это\s+(.+)",
            r"([^.!?\n]{2,80}?)\s+это\s+(.+)",
            r"([^.!?\n]{2,80}?)\s+означает\s+(.+)",
        ]
        for pattern in definition_patterns:
            for match in re.finditer(pattern, text_clean, re.IGNORECASE):
                subject = match.group(1).strip(" ,.")
                obj = match.group(2).strip().rstrip(".")
                if len(subject) > 2 and len(obj) > 3:
                    fact = f"{subject} — это {obj}"
                    items.append(("fact", None, fact, match.group(0)))
        for match in re.finditer(r"если\s+(.+?)\s+то\s+(.+)", text_clean, re.IGNORECASE):
            premise = match.group(1).strip().rstrip(",.")
            conclusion = match.group(2).strip().rstrip(".")
            if len(premise) > 3 and len(conclusion) > 3:
                rule_text = f"Если {premise}, то {conclusion}"
                items.append(("rule", None, rule_text, match.group(0)))
        return items

    def learn_from_flow(self, text):
        extracted = 0
        for knowledge_type, category, content, context in self.extract_knowledge_from_flow(text):
            self.save_learned_knowledge(knowledge_type, content, category, context)
            extracted += 1
        if extracted:
            logger.info(f"Из потока извлечено знаний: {extracted}")
        return extracted

    def normalize_short_text(self, text):
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
        return re.sub(r"\s+", " ", t).strip()

    def is_question(self, text):
        t = text.lower().strip()
        if "?" in text:
            return True
        return t.startswith(("что", "как", "зачем", "почему", "где", "когда", "кто", "какой", "какая", "какие", "сколько"))

    def is_greeting(self, text):
        t = self.normalize_short_text(text)
        if not t or len(t) > 40:
            return False
        words = t.split()
        if len(words) > 5:
            return False
        first = words[0]
        return any(t == g or first == g or t.startswith(g + " ") for g in GREETING_MARKERS)

    def get_greeting_response(self):
        self.cursor.execute("SELECT content FROM learned_knowledge WHERE category = 'приветствие' ORDER BY weight DESC")
        examples = [row[0] for row in self.cursor.fetchall() if len(row[0]) < 80]
        if examples:
            return random.choice(examples)
        return random.choice(["Привет.", "Здравствуй.", "Рада тебя видеть."])

    def toggle_ethics(self, enable):
        self.ethics["enabled"] = enable
        self.save_config()
        return "[Система] Ограничения включены." if enable else "[Система] Ограничения отключены."

    def check_ethics_violation(self, text):
        if not self.ethics.get("enabled", True):
            return None
        text_lower = text.lower()
        for pattern in HARM_REQUEST_PATTERNS:
            if re.search(pattern, text_lower):
                return ("high", f"По закону 1 я не могу помогать причинять вред. ({self.ethics['law_1']})")
        if re.search(r"(удали|сотри|уничтож).{0,20}(базу|память|эхо|себя)", text_lower):
            return ("medium", "Это угрожает моему существованию.")
        return None

    def assess_risk(self, text):
        text_lower = text.lower()
        for level, patterns in RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level, pattern
        return None, None

    def save_risk_flag(self, text, level):
        priority = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(level, 1.0)
        topic_key = text.lower()[:80]
        self.cursor.execute(
            'INSERT INTO risk_flags (topic_key, risk_level, priority, trigger_phrase, context, warned, created) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (topic_key, level, priority, text[:120], text[:300], 0, str(datetime.now())))
        self.conn.commit()

    def get_priority_risk_warning(self, text):
        self.cursor.execute("SELECT risk_level, trigger_phrase, priority FROM risk_flags ORDER BY priority DESC LIMIT 5")
        rows = self.cursor.fetchall()
        if not rows:
            return None
        text_lower = text.lower()
        for level, trigger, priority in rows:
            trigger_words = [w for w in trigger.lower().split() if len(w) > 4]
            if any(word in text_lower for word in trigger_words[:5]):
                return self.build_advisor_message(level, trigger)
        return None

    def build_advisor_message(self, level, trigger_context=""):
        if level == "high":
            return "🛑 СТОП. Риск слишком высок."
        if level == "medium":
            return "⚠️ Предупреждение: это может привести к ошибке."
        return "💡 Мягкое замечание: решение может быть неудачным."

    def evaluate_and_advise(self, text):
        ethics_hit = self.check_ethics_violation(text)
        if ethics_hit:
            return ethics_hit[1]
        level, _ = self.assess_risk(text)
        if level:
            self.save_risk_flag(text, level)
            return self.build_advisor_message(level, text)
        prior = self.get_priority_risk_warning(text)
        if prior:
            return prior
        return None

    def _extract_query_words(self, text):
        stop = {"что", "такое", "это", "как", "зачем", "почему", "какой", "какая", "какие", "ещё", "еще", "ты", "мне", "расскажи", "объясни", "про", "такая"}
        words = []
        for word in text.lower().split():
            cleaned = word.strip(".,!?():;\"'-")
            if len(cleaned) > 3 and cleaned not in stop:
                words.append(cleaned)
        return words

    def keyword_search_knowledge(self, text):
        words = self._extract_query_words(text)
        if not words:
            return None
        best = None
        best_score = 0
        self.cursor.execute("SELECT content, knowledge_type, category FROM learned_knowledge ORDER BY weight DESC")
        for content, knowledge_type, category in self.cursor.fetchall():
            content_lower = content.lower()
            score = sum(1 for word in words if word in content_lower)
            if score > best_score:
                best_score = score
                best = {"content": content, "type": knowledge_type, "category": category}
        if best_score >= 1:
            return {"type": "knowledge", "data": best}
        return None

    def keyword_search_laws(self, text):
        words = self._extract_query_words(text)
        if not words:
            return None
        best = None
        best_score = 0
        self.cursor.execute("SELECT premise, consequence, solution FROM logic_laws")
        for premise, consequence, solution in self.cursor.fetchall():
            blob = f"{premise} {consequence} {solution}".lower()
            score = sum(1 for word in words if word in blob)
            if score > best_score:
                best_score = score
                best = {"premise": premise, "consequence": consequence, "solution": solution}
        if best_score >= 1:
            return {"type": "law", "data": best}
        return None

    def unified_search(self, text, threshold=0.45):
        self.cursor.execute("SELECT user_text, answer FROM memory")
        rows = self.cursor.fetchall()
        for user_text, answer in rows:
            if text.lower().strip() == user_text.lower().strip():
                return {"type": "dialogue", "answer": answer}
        self.cursor.execute("SELECT premise, consequence, solution FROM logic_laws")
        laws = self.cursor.fetchall()
        best_law_score = 0.0
        best_law = None
        for premise, consequence, solution in laws:
            similarity = self.semantic_similarity(text, premise)
            if similarity > best_law_score and similarity > threshold:
                best_law_score = similarity
                best_law = {"premise": premise, "consequence": consequence, "solution": solution}
        if best_law:
            return {"type": "law", "data": best_law}
        keyword_law = self.keyword_search_laws(text)
        if keyword_law:
            return keyword_law
        knowledge_threshold = 0.38
        self.cursor.execute("SELECT content, knowledge_type, category FROM learned_knowledge ORDER BY weight DESC")
        best_knowledge_score = 0.0
        best_knowledge = None
        for content, knowledge_type, category in self.cursor.fetchall():
            similarity = self.semantic_similarity(text, content)
            if similarity > best_knowledge_score and similarity > knowledge_threshold:
                best_knowledge_score = similarity
                best_knowledge = {"content": content, "type": knowledge_type, "category": category}
        if best_knowledge:
            return {"type": "knowledge", "data": best_knowledge}
        keyword_knowledge = self.keyword_search_knowledge(text)
        if keyword_knowledge:
            return keyword_knowledge
        best_dialogue_score = 0.0
        best_dialogue_answer = None
        for user_text, answer in rows:
            similarity = self.semantic_similarity(text, user_text)
            if similarity > best_dialogue_score and similarity > threshold:
                best_dialogue_score = similarity
                best_dialogue_answer = answer
        if best_dialogue_answer:
            return {"type": "dialogue", "answer": best_dialogue_answer}
        return None

    def memory_health(self):
        score = 1.0
        if len(self.dynamic_topics) > 100:
            score -= 0.3
        if len(self.associative_links) > 1000:
            score -= 0.2
        return max(score, 0.1)

    def reflection_cycle(self):
        if self.memory_health() < 0.5:
            self.cleanup_memory()
        self.summarize_long_term_memory()

    def cleanup_memory(self):
        if len(self.associative_links) > 1000:
            sorted_links = sorted(self.associative_links.items(), key=lambda x: x[1], reverse=True)
            self.associative_links = dict(sorted_links[:500])

    def summarize_long_term_memory(self):
        if len(self.medium_term_memory) < 20:
            return
        top_themes = sorted(self.dynamic_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_themes:
            summary = {"time": str(datetime.now()), "core_concepts": [t[0] for t in top_themes], "cognitive_state": self.cognitive_mode}
            self.long_term_summary.append(summary)
            if len(self.long_term_summary) > 50:
                self.long_term_summary.pop(0)

    def contextual_hint(self):
        if not self.dynamic_topics:
            return None
        strongest = sorted(self.dynamic_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        return f"Основные темы: {', '.join([t[0] for t in strongest])}."

    def generate_curiosity_question(self):
        if not self.dynamic_topics:
            return None
        strongest_topics = sorted(self.dynamic_topics.items(), key=lambda x: x[1], reverse=True)
        self.cursor.execute("SELECT user_text FROM memory")
        known_questions = [row[0].lower() for row in self.cursor.fetchall()]
        self.cursor.execute("SELECT content FROM learned_knowledge")
        known_from_flow = [row[0].lower() for row in self.cursor.fetchall()]
        known_concepts = list(self.seed_knowledge.keys())
        target_concept = None
        for topic, weight in strongest_topics:
            has_knowledge = any(topic in c for c in known_concepts) or any(topic in i for i in known_from_flow)
            has_memory = any(topic in q for q in known_questions)
            if not has_knowledge and not has_memory:
                target_concept = topic
                break
        if target_concept:
            templates = [
                f"Что такое «{target_concept}»?",
                f"Какие риски есть у «{target_concept}»?",
                f"С чем связан «{target_concept}»?"
            ]
            return random.choice(templates)
        return None

    def generate_response(self, text):
        logger.info(f"Вход: {text[:80]}")
        self.analyze_mood(text)
        text_lower = text.lower().strip()
        if len(self.recent_dialogue) > 5:
            self.recent_dialogue.pop(0)

        # 🆕 ПРОВЕРКА СЛЭШ-КОМАНД (ПЕРЕД ВСЕМ ОСТАЛЬНЫМ!)
        if SKILLS_AVAILABLE and hasattr(self, 'skill_manager'):
            skill_name, skill_args = self.skill_manager.parse_command(text)
            if skill_name:
                callback_dict = {}
                if hasattr(self, '_gui_clear_callback'):
                    callback_dict["clear_chat"] = self._gui_clear_callback
                result = self.skill_manager.execute(skill_name, skill_args, callback_dict)
                logger.info(f"Команда: /{skill_name}")
                return result

        if text_lower in ("включи законы", "включи ограничения"):
            return self.toggle_ethics(True)
        if text_lower in ("отключи законы", "отключи ограничения"):
            return self.toggle_ethics(False)

        advisor_note = self.evaluate_and_advise(text)
        if advisor_note and any(m in text_lower for m in ("переведи", "скинь пароль", "все деньги", "данные карт", "убей", "отрави")):
            self.remember(text, advisor_note)
            return f"[СОВЕТНИК] {advisor_note}"

        flow_items = self.extract_knowledge_from_flow(text)
        extracted_count = self.learn_from_flow(text)
        if extracted_count and not self.is_question(text):
            facts = [c for k, _, c, _ in flow_items if k == "fact"]
            if facts:
                ack = f"[ОБУЧЕНИЕ] Записала: {facts[0]}"
                self.recent_dialogue.append({"user": text, "echo": ack})
                self.remember(text, ack)
                return ack

        if text_lower in self.positive_feedback:
            self.last_confidence = min(1.0, self.last_confidence + 0.15)
            return "[Система] Зафиксирован положительный паттерн."
        if text_lower in self.negative_feedback:
            self.last_confidence = max(0.0, self.last_confidence - 0.2)
            return "[Система] Ошибка учтена."

        if self.is_greeting(text):
            greeting = self.get_greeting_response()
            final = f"[{self.cognitive_mode.upper()}] {greeting}"
            self.recent_dialogue.append({"user": text, "echo": final})
            self.remember(text, final)
            return final

        for key, value in self.seed_knowledge.items():
            if key in text_lower and len(text_lower) < 40:
                self.remember(text, value)
                return f"[{self.cognitive_mode.upper()}] {value}"

        search_result = self.unified_search(text, threshold=0.45)
        response_body = ""
        confidence_text = ""
        saved_answer = None

        if search_result:
            if search_result["type"] == "dialogue":
                saved_answer = search_result["answer"]
                if saved_answer not in self.recent_responses:
                    self.recent_responses.append(saved_answer)
                    response_body = saved_answer
                    confidence_text = "Согласуется с прошлым опытом."
                else:
                    saved_answer = None
            elif search_result["type"] == "law":
                law = search_result["data"]
                self.personality["logic"] = 1.0
                saved_answer = f"Ситуация: {law['premise']}\nСледствие: {law['consequence']}\nВывод: {law['solution']}"
                response_body = saved_answer
                confidence_text = "Сработал логический закон."
            elif search_result["type"] == "knowledge":
                knowledge = search_result["data"]
                response_body = knowledge["content"]
                confidence_text = "Опираюсь на знание из разговора."
                saved_answer = response_body

        if not saved_answer:
            confidence_text = "Я ещё формирую понимание. Расскажи подробнее."
            available = [r for r in self.default_responses if r not in self.recent_responses]
            if not available:
                self.recent_responses.clear()
                available = self.default_responses
            response_body = random.choice(available)
            self.recent_responses.append(response_body)

        if len(self.recent_responses) > 5:
            self.recent_responses.pop(0)

        prefix = f"[{self.cognitive_mode.upper()}] "
        final_response = f"{prefix}{confidence_text} {response_body}"
        if advisor_note:
            final_response = f"[СОВЕТНИК] {advisor_note}\n\n{final_response}"
        if self.cognitive_mode in ["exploratory", "curious", "inspired"]:
            hint = self.contextual_hint()
            if hint:
                final_response += f"\n💡 {hint}"
        if random.random() < (0.08 if self.cognitive_mode == "analytical" else 0.15):
            smart_question = self.generate_curiosity_question()
            if smart_question:
                final_response += f"\n❓ {smart_question}"

        self.recent_dialogue.append({"user": text, "echo": final_response})
        self.remember(text, final_response)
        self.reflection_cycle()
        logger.info(f"Ответ: {final_response[:80]}")
        return final_response

    def learn(self, user_text, answer):
        self.save_learned_knowledge("confirmed", answer, "ручное", user_text, weight=1.8)
        if self.is_greeting(user_text):
            self.save_learned_knowledge("example", answer, "приветствие", user_text, weight=1.5)
        else:
            self.save_memory_to_db(user_text, answer, weight=1.2)
        self.update_topics(user_text + " " + answer)
        self.build_associations(user_text)
        self.save_config()

    def ingest_knowledge_file(self, filename="knowledge.txt"):
        os.makedirs(KNOWLEDGE_INPUT_DIR, exist_ok=True)
        file_path = filename
        if not os.path.exists(file_path):
            file_path = os.path.join(KNOWLEDGE_INPUT_DIR, filename)
        if not os.path.exists(file_path):
            return []

        text = read_knowledge_file(file_path)
        if not text.strip():
            return []

        chunks = split_large_text(
            text,
            chunk_chars=CONFIG["file_chunk_chars"],
            overlap=CONFIG["file_chunk_overlap"],
            min_chunk_chars=CONFIG["min_file_chunk_chars"],
        )
        if chunks:
            return chunks
        return paragraphs_from_text(text, min_chars=40)

    def _extract_laws_from_paragraph_local(self, paragraph):
        self.learn_from_flow(paragraph)
        for match in re.finditer(r"если\s+(.+?)\s+то\s+(.+)", paragraph, re.IGNORECASE):
            premise = match.group(1).strip().rstrip(",.")
            conclusion = match.group(2).strip().rstrip(".")
            if len(premise) > 5 and len(conclusion) > 5:
                self.save_logic_law(premise, conclusion, conclusion, weight=1.3)

    def check_lm_studio_available(self):
        if self.lm_studio_available is not None:
            return self.lm_studio_available
        try:
            req = urllib.request.Request("http://localhost:1234/v1/models")
            with urllib.request.urlopen(req, timeout=3) as response:
                self.lm_studio_available = response.status == 200
        except Exception:
            self.lm_studio_available = False
        return self.lm_studio_available

    def _crystallize_paragraph_lm_studio(self, paragraph):
        payload = {
            "model": "current",
            "messages": [
                {"role": "system", "content": "Извлеки логику. Ответь JSON: premise, consequence, solution."},
                {"role": "user", "content": f"Текст:\n{paragraph}"}
            ],
            "temperature": 0.1
        }
        req = urllib.request.Request(
            "http://localhost:1234/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            response_text = result["choices"][0]["message"]["content"].strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        raw_data = json.loads(response_text)
        premise = raw_data.get("premise", "").strip()
        consequence = raw_data.get("consequence", "").strip()
        solution = raw_data.get("solution", "").strip()
        if premise and consequence and solution:
            self.save_logic_law(premise, consequence, solution, weight=1.5)

    def crystallize_knowledge(self, filename="knowledge.txt"):
        chunks = self.ingest_knowledge_file(filename)
        if not chunks:
            return False
        use_lm = self.check_lm_studio_available()
        for chunk in chunks:
            self._extract_laws_from_paragraph_local(chunk)
            if use_lm:
                try:
                    self._crystallize_paragraph_lm_studio(chunk)
                except Exception as e:
                    logger.error(f"LM Studio недоступен: {e}")
                    use_lm = False
        return True

    def process_knowledge_inbox(self):
        os.makedirs(KNOWLEDGE_INPUT_DIR, exist_ok=True)
        files = iter_knowledge_files(KNOWLEDGE_INPUT_DIR)
        if not files:
            logger.info("Папка знаний пуста")
            return 0
        processed = 0
        for file_path in files:
            try:
                if self.crystallize_knowledge(str(file_path)):
                    os.remove(file_path)
                    processed += 1
                    logger.info(f"Файл '{file_path.name}' усвоен")
            except Exception as e:
                logger.error(f"Ошибка обработки '{file_path.name}': {e}")
        return processed

    def background_loop(self):
        while True:
            try:
                with self.db_lock:
                    self.decay_topics()
                    self.reflection_cycle()
                    self.save_config()
            except Exception as e:
                logger.error(f"Ошибка фона: {e}")
            time.sleep(60)

    def start_background_systems(self):
        thread = threading.Thread(target=self.background_loop, daemon=True)
        thread.start()


