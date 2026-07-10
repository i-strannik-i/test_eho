import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from .core import LOG_FILE, SKILLS_AVAILABLE, SYSTEM_VERSION, UnifiedAssistant, logger
from project_paths import LORA_OUTPUT_DIR, TEACHER_DATA_FILE, TRAINING_DATA_FILE


class AssistantGUI:
    def __init__(self, root):
        self.assistant = UnifiedAssistant()
        self.root = root
        self.skills_window = None
        self.primary_skills = self._collect_primary_skills()
        self.message_in_flight = False
        self.learning_mode_enabled = False
        self.ui_queue = queue.Queue()
        self.updating_weight_controls = False
        self.weight_vars = {}
        self.weight_value_labels = {}
        self.learn_sidebar_button = None
        self.learn_quick_button = None
        self.dataset_sidebar_button = None
        self.dataset_quick_button = None
        self.training_sidebar_button = None
        self.training_quick_button = None
        self.background_job_process = None
        self.background_job_reader = None
        self.background_job_kind = None
        self.background_job_title = ""
        self.background_job_started_at = None
        self.background_job_stopping = False
        self.pipeline_status = self._build_idle_pipeline_status()
        self.dataset_snapshot = self._collect_dataset_snapshot()
        self.sash_ratios = {
            "workspace_split": 0.72,
            "chat_split": 0.5,
            "right_panel_split": 0.34,
        }
        self._pending_sash_updates = set()

        assistant_name = self.assistant.personality.get("name", "Эхо")
        self.root.title(f"{SYSTEM_VERSION} — {assistant_name}")
        self.root.geometry("1180x760")
        self.root.minsize(760, 500)
        self.root.configure(bg="#0f141a")
        self.root.protocol("WM_DELETE_WINDOW", self._handle_window_close)

        self._build_layout()
        self.setup_global_shortcuts()

        if SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager"):
            def _append_skill_log(message):
                self._queue_ui("right", f"{message}\n")

            self.assistant.skill_manager.register_gui(_append_skill_log)

            def _clear_chat():
                self._queue_ui("clear", None)

            self.assistant._gui_clear_callback = _clear_chat

        self._insert_boot_message()
        self.refresh_weight_panel()
        self.refresh_runtime_status()
        self._process_ui_queue()
        logger.info("GUI инициализирован")
        self.start_proactive_pulse()

    def _build_layout(self):
        self.app_shell = tk.Frame(self.root, bg="#0f141a")
        self.app_shell.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.app_shell, bg="#151d26", width=118)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.workspace = tk.Frame(self.app_shell, bg="#0f141a")
        self.workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_sidebar()
        self._build_header()
        self._build_workspace_split()
        self._build_chat_panel()
        self._build_composer()
        self._update_learning_mode_buttons()
        self._update_background_job_buttons()
        self.root.after_idle(self._initialize_responsive_sashes)

    def _build_workspace_split(self):
        self.workspace_split = tk.PanedWindow(
            self.workspace,
            orient=tk.VERTICAL,
            sashwidth=10,
            bd=0,
            bg="#0f141a",
            relief=tk.FLAT,
        )
        self.workspace_split.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 16))

        self.chat_container = tk.Frame(self.workspace_split, bg="#0f141a")
        self.composer_container = tk.Frame(self.workspace_split, bg="#0f141a")
        self.workspace_split.add(self.chat_container, stretch="always", minsize=260)
        self.workspace_split.add(self.composer_container, minsize=150)
        self._register_paned_window("workspace_split", self.workspace_split, tk.VERTICAL)

    def _build_sidebar(self):
        assistant_name = self.assistant.personality.get("name", "Эхо")
        brand = tk.Frame(self.sidebar, bg="#1b2632", padx=10, pady=12)
        brand.pack(fill=tk.X, padx=12, pady=(14, 10))

        tk.Label(
            brand,
            text="ЭХО",
            font=("Segoe UI", 16, "bold"),
            bg="#1b2632",
            fg="#f7fafc",
        ).pack()
        tk.Label(
            brand,
            text=assistant_name,
            font=("Segoe UI", 9),
            bg="#1b2632",
            fg="#8fb4d8",
        ).pack(pady=(2, 0))

        buttons = [
            ("🚀", "Учиться", self.toggle_learning_mode, False),
            ("🗂", "Датасет", self.prepare_dataset, False),
            ("▶", "Запуск", self.start_training_pipeline, False),
            ("📚", "Знания", self.start_file_learning, False),
            ("🧠", "Ручное", self.teach_response, False),
            ("📊", "Стат", lambda: self.queue_command("/стат", send=True), False),
            ("✨", "Скиллы", self.open_skills_window, False),
            ("🧹", "Чисто", self.clear_chat, False),
            ("■", "Стоп", self.interrupt_active_task, True),
        ]

        for icon, label, command, danger in buttons:
            button = self._create_sidebar_button(icon, label, command, danger=danger)
            if label == "Учиться":
                self.learn_sidebar_button = button
            elif label == "Датасет":
                self.dataset_sidebar_button = button
            elif label == "Запуск":
                self.training_sidebar_button = button

        footer = tk.Label(
            self.sidebar,
            text="Ctrl+C/V\nCtrl+Z стоп",
            justify=tk.CENTER,
            font=("Segoe UI", 9),
            bg="#151d26",
            fg="#7f95ab",
        )
        footer.pack(side=tk.BOTTOM, pady=16)

    def _build_header(self):
        self.header = tk.Frame(self.workspace, bg="#0f141a")
        self.header.pack(fill=tk.X, padx=18, pady=(18, 10))

        title_block = tk.Frame(self.header, bg="#0f141a")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            title_block,
            text="Двухпанельная консоль Эхо",
            font=("Segoe UI", 18, "bold"),
            bg="#0f141a",
            fg="#f5f7fa",
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Слева учитель Ollama, справа ученик Эхо, общий ввод снизу.",
            font=("Segoe UI", 10),
            bg="#0f141a",
            fg="#8aa2b8",
        ).pack(anchor="w", pady=(3, 0))

        info_panel = tk.Frame(self.header, bg="#111923", padx=14, pady=10)
        info_panel.pack(side=tk.RIGHT)

        self.mode_label = tk.Label(info_panel, font=("Segoe UI", 9, "bold"), bg="#111923", fg="#9fd3ff")
        self.mode_label.pack(anchor="e")
        self.task_label = tk.Label(info_panel, font=("Segoe UI", 9), bg="#111923", fg="#c9d4df")
        self.task_label.pack(anchor="e", pady=(4, 0))
        self.progress_label = tk.Label(
            info_panel,
            font=("Segoe UI", 9),
            bg="#111923",
            fg="#88c8a7",
            wraplength=360,
            justify=tk.RIGHT,
        )
        self.progress_label.pack(anchor="e", pady=(4, 0))
        self.dataset_label = tk.Label(
            info_panel,
            font=("Consolas", 8),
            bg="#111923",
            fg="#7f95ab",
            wraplength=360,
            justify=tk.RIGHT,
        )
        self.dataset_label.pack(anchor="e", pady=(6, 0))

    def _build_chat_panel(self):
        chat_outer = tk.Frame(self.chat_container, bg="#26313c", bd=0, padx=1, pady=1)
        chat_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        split = tk.PanedWindow(
            chat_outer,
            orient=tk.HORIZONTAL,
            sashwidth=8,
            bd=0,
            bg="#26313c",
            relief=tk.FLAT,
        )
        split.pack(fill=tk.BOTH, expand=True)
        self.chat_split = split

        left_panel = tk.Frame(split, bg="#0f141a")
        right_panel = tk.Frame(split, bg="#0f141a")
        split.add(left_panel, stretch="always", minsize=220)
        split.add(right_panel, stretch="always", minsize=220)
        self._register_paned_window("chat_split", self.chat_split, tk.HORIZONTAL)

        self.left_chat_area = self._create_output_panel(
            left_panel,
            f"Ollama ({self.assistant.ollama_model})",
            "#7fd0ff",
            "#101821",
        )
        self.right_chat_area = self._create_output_panel(
            right_panel,
            "Нейросеть / Эхо",
            "#a4e7b3",
            "#121922",
            with_weights=True,
        )

    def _create_output_panel(self, parent, title, accent, text_bg, with_weights=False):
        shell = tk.Frame(parent, bg="#26313c", bd=0, padx=1, pady=1)
        shell.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        inner = tk.Frame(shell, bg="#0f141a")
        inner.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(inner, bg="#16202b", padx=12, pady=10)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg="#16202b",
            fg=accent,
        ).pack(anchor="w")

        content_host = inner
        if with_weights:
            content_split = tk.PanedWindow(
                inner,
                orient=tk.VERTICAL,
                sashwidth=8,
                bd=0,
                bg="#0f141a",
                relief=tk.FLAT,
            )
            content_split.pack(fill=tk.BOTH, expand=True)
            self.right_panel_split = content_split
            weights_host = tk.Frame(content_split, bg="#10161d")
            chat_host = tk.Frame(content_split, bg=text_bg)
            content_split.add(weights_host, minsize=110)
            content_split.add(chat_host, stretch="always", minsize=120)
            self._register_paned_window("right_panel_split", self.right_panel_split, tk.VERTICAL)
            self._build_weights_panel(weights_host)
            content_host = chat_host

        chat_area = scrolledtext.ScrolledText(
            content_host,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg=text_bg,
            fg="#d7e0e9",
            insertbackground="#ffffff",
            bd=0,
            highlightthickness=0,
            padx=14,
            pady=14,
        )
        chat_area.pack(fill=tk.BOTH, expand=True)
        self.setup_context_menu(chat_area, allow_paste=False, allow_cut=False)
        return chat_area

    def _build_weights_panel(self, parent):
        panel = tk.Frame(parent, bg="#10161d", padx=12, pady=10)
        panel.pack(fill=tk.X)

        top = tk.Frame(panel, bg="#10161d")
        top.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            top,
            text="Веса ученика",
            font=("Segoe UI", 10, "bold"),
            bg="#10161d",
            fg="#c9f3cf",
        ).pack(side=tk.LEFT)
        self.weight_mode_label = tk.Label(
            top,
            text="",
            font=("Segoe UI", 9),
            bg="#10161d",
            fg="#8ec0df",
        )
        self.weight_mode_label.pack(side=tk.RIGHT)

        self.weight_change_label = tk.Label(
            panel,
            text="",
            font=("Consolas", 9),
            bg="#10161d",
            fg="#8aa2b8",
            wraplength=520,
            justify=tk.LEFT,
            anchor="w",
        )
        self.weight_change_label.pack(fill=tk.X, pady=(0, 8))

        rows = [
            ("energy", "Energy"),
            ("logic", "Logic"),
            ("creativity", "Creativity"),
            ("stability", "Stability"),
            ("curiosity", "Curiosity"),
        ]
        for key, title in rows:
            row = tk.Frame(panel, bg="#10161d")
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row,
                text=title,
                width=11,
                anchor="w",
                font=("Segoe UI", 9, "bold"),
                bg="#10161d",
                fg="#d7e0e9",
            ).pack(side=tk.LEFT)

            value_label = tk.Label(
                row,
                text="0.00",
                width=5,
                font=("Consolas", 9),
                bg="#10161d",
                fg="#9fd3ff",
            )
            value_label.pack(side=tk.RIGHT)
            self.weight_value_labels[key] = value_label

            var = tk.DoubleVar(value=self.assistant.personality.get(key, 0.0))
            slider = tk.Scale(
                row,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                variable=var,
                showvalue=False,
                highlightthickness=0,
                bd=0,
                sliderrelief=tk.FLAT,
                troughcolor="#203041",
                bg="#10161d",
                fg="#d7e0e9",
                activebackground="#7fd0ff",
                length=250,
                command=lambda value, weight_key=key: self._on_weight_slider(weight_key, value),
            )
            slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 8))
            self.weight_vars[key] = var

        controls = tk.Frame(panel, bg="#10161d")
        controls.pack(fill=tk.X, pady=(8, 0))
        self._create_quick_button(
            controls,
            "Сбросить веса",
            self.reset_student_weights,
        ).pack(side=tk.LEFT)

    def _on_weight_slider(self, key, value):
        if self.updating_weight_controls:
            return
        try:
            numeric = float(value)
        except ValueError:
            return
        with self.assistant.db_lock:
            self.assistant.set_personality_weight(key, numeric)
        self.refresh_weight_panel()

    def reset_student_weights(self):
        with self.assistant.db_lock:
            self.assistant.reset_personality_weights()
        self.refresh_weight_panel()
        self._append_right("[Система] Веса ученика сброшены к базовым значениям.\n\n")

    def _build_composer(self):
        composer = tk.Frame(self.composer_container, bg="#0f141a")
        composer.pack(fill=tk.BOTH, expand=True)

        quick_actions = tk.Frame(composer, bg="#0f141a")
        quick_actions.pack(fill=tk.X, pady=(0, 8))

        self.learn_quick_button = self._create_quick_button(
            quick_actions,
            "Учиться",
            self.toggle_learning_mode,
        )
        self.learn_quick_button.pack(side=tk.LEFT, padx=(0, 8))
        self.dataset_quick_button = self._create_quick_button(
            quick_actions,
            "Подготовить датасет",
            self.prepare_dataset,
        )
        self.dataset_quick_button.pack(side=tk.LEFT, padx=(0, 8))
        self.training_quick_button = self._create_quick_button(
            quick_actions,
            "Запуск обучения",
            self.start_training_pipeline,
        )
        self.training_quick_button.pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Знания", self.start_file_learning).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Ручное", self.teach_response).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Скиллы", self.open_skills_window).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Стат", lambda: self.queue_command("/стат", send=True)).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Стоп", self.interrupt_active_task, danger=True).pack(side=tk.RIGHT)
        self._create_quick_button(quick_actions, "Отправить", self.send_message, accent=True).pack(side=tk.RIGHT, padx=(0, 8))

        entry_shell = tk.Frame(composer, bg="#26313c", bd=0, padx=1, pady=1)
        entry_shell.pack(fill=tk.X)

        entry_inner = tk.Frame(entry_shell, bg="#121a24")
        entry_inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            entry_inner,
            text="Поле ввода  |  Enter отправить  |  Shift+Enter новая строка  |  Ctrl+Z остановить задачу",
            font=("Segoe UI", 9),
            bg="#121a24",
            fg="#7f95ab",
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(10, 6))

        self.entry = tk.Text(
            entry_inner,
            font=("Consolas", 12),
            height=5,
            wrap=tk.WORD,
            undo=True,
            bg="#121a24",
            fg="#ffffff",
            insertbackground="white",
            bd=0,
            highlightthickness=0,
            padx=12,
            pady=6,
        )
        self.entry.pack(fill=tk.BOTH, expand=True)
        self.entry.bind("<Return>", self.handle_enter)
        self.entry.bind("<Shift-Return>", self.insert_newline)
        self.setup_context_menu(self.entry, allow_paste=True, allow_cut=True)

    def _create_sidebar_button(self, icon, label, command, danger=False):
        bg = "#202d3b" if not danger else "#512127"
        active_bg = "#2c3d4f" if not danger else "#743039"
        button = tk.Button(
            self.sidebar,
            text=f"{icon}\n{label}",
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg=bg,
            fg="#f4f7fb",
            activebackground=active_bg,
            activeforeground="#ffffff",
            bd=0,
            relief=tk.FLAT,
            padx=6,
            pady=10,
            justify=tk.CENTER,
            cursor="hand2",
        )
        button.pack(fill=tk.X, padx=12, pady=5)
        return button

    def _create_quick_button(self, parent, text, command, accent=False, danger=False):
        bg = "#1b2632"
        active_bg = "#2b3949"
        if accent:
            bg = "#1f5f8b"
            active_bg = "#2875aa"
        elif danger:
            bg = "#6d2b33"
            active_bg = "#8a3742"
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg=bg,
            fg="#f7fbff",
            activebackground=active_bg,
            activeforeground="#ffffff",
            bd=0,
            relief=tk.FLAT,
            padx=14,
            pady=7,
            cursor="hand2",
        )

    def _update_learning_mode_buttons(self):
        enabled = self.learning_mode_enabled

        if self.learn_sidebar_button:
            self.learn_sidebar_button.config(
                text="🚀\nУчиться\nВКЛ" if enabled else "🚀\nУчиться\nВЫКЛ",
                bg="#1f6f43" if enabled else "#202d3b",
                activebackground="#2b9259" if enabled else "#2c3d4f",
            )

        if self.learn_quick_button:
            self.learn_quick_button.config(
                text="Учиться: ВКЛ" if enabled else "Учиться: ВЫКЛ",
                bg="#1d6a47" if enabled else "#1b2632",
                activebackground="#2a8e60" if enabled else "#2b3949",
            )

    def _update_background_job_buttons(self):
        dataset_active = self._background_job_running() and self.background_job_kind == "dataset"
        training_active = self._background_job_running() and self.background_job_kind == "training"

        if self.dataset_sidebar_button:
            self.dataset_sidebar_button.config(
                text="🗂\nДатасет\nИДЁТ" if dataset_active else "🗂\nДатасет",
                bg="#6a5320" if dataset_active else "#202d3b",
                activebackground="#8b6c28" if dataset_active else "#2c3d4f",
            )
        if self.training_sidebar_button:
            self.training_sidebar_button.config(
                text="▶\nЗапуск\nИДЁТ" if training_active else "▶\nЗапуск",
                bg="#1f5f8b" if training_active else "#202d3b",
                activebackground="#2875aa" if training_active else "#2c3d4f",
            )
        if self.dataset_quick_button:
            self.dataset_quick_button.config(
                text="Датасет: ИДЁТ" if dataset_active else "Подготовить датасет",
                bg="#6a5320" if dataset_active else "#1b2632",
                activebackground="#8b6c28" if dataset_active else "#2b3949",
            )
        if self.training_quick_button:
            self.training_quick_button.config(
                text="Обучение: ИДЁТ" if training_active else "Запуск обучения",
                bg="#1f5f8b" if training_active else "#1b2632",
                activebackground="#2875aa" if training_active else "#2b3949",
            )

    def toggle_learning_mode(self):
        self.learning_mode_enabled = not self.learning_mode_enabled
        self._update_learning_mode_buttons()

        if self.learning_mode_enabled:
            message = (
                "[Система] Режим обучения включён. "
                "После каждого нового ответа учителя ученик автоматически усвоит урок.\n\n"
            )
        else:
            message = "[Система] Режим обучения выключен. Автопередача уроков остановлена.\n\n"

        self._append_left(message)
        self._append_right(message)

    def _build_idle_pipeline_status(self):
        return {
            "active": False,
            "kind": None,
            "title": "нет фоновых задач",
            "stage": "Ожидание",
            "summary": "Прогресс: ожидание",
            "details": "",
            "last_line": "",
            "dataset_rows": 0,
            "source_counts": {},
            "checkpoint": "нет",
            "loss": "нет",
            "epoch": "нет",
        }

    def _count_jsonl_rows(self, path):
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                return sum(1 for line in file_handle if line.strip())
        except OSError:
            return 0

    def _latest_checkpoint_name(self):
        checkpoint_root = Path(LORA_OUTPUT_DIR)
        if not checkpoint_root.exists():
            return "нет"
        checkpoints = []
        for path in checkpoint_root.iterdir():
            if not path.is_dir() or not path.name.startswith("checkpoint-"):
                continue
            try:
                order = int(path.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((order, path.name))
        if not checkpoints:
            return "нет"
        checkpoints.sort()
        return checkpoints[-1][1]

    def _collect_dataset_snapshot(self):
        return {
            "teacher_lessons": self.assistant.count_teacher_lessons(),
            "teacher_dataset": TEACHER_DATA_FILE,
            "train_data_exists": Path(TRAINING_DATA_FILE).exists(),
            "train_data_rows": self._count_jsonl_rows(TRAINING_DATA_FILE),
            "latest_checkpoint": self._latest_checkpoint_name(),
        }

    def _format_dataset_snapshot(self):
        snapshot = self.dataset_snapshot
        dataset_text = (
            f"dataset {snapshot['train_data_rows']}"
            if snapshot["train_data_exists"]
            else "dataset нет"
        )
        source_bits = [
            f"уроки {snapshot['teacher_lessons']}",
            dataset_text,
            f"ckpt {snapshot['latest_checkpoint']}",
        ]
        dynamic_sources = self.pipeline_status.get("source_counts") or {}
        if dynamic_sources:
            ordered_keys = [
                "teacher_lessons",
                "dialogues",
                "knowledge_base",
                "knowledge_files",
                "logic_laws",
                "hf_yagpt",
                "hf_alice",
            ]
            labels = {
                "teacher_lessons": "teacher",
                "dialogues": "dialogues",
                "knowledge_base": "knowledge",
                "knowledge_files": "files",
                "logic_laws": "logic",
                "hf_yagpt": "yagpt",
                "hf_alice": "alice",
            }
            parts = []
            for key in ordered_keys:
                value = dynamic_sources.get(key)
                if value:
                    parts.append(f"{labels[key]} {value}")
            if parts:
                source_bits.append(" | ".join(parts[:4]))
        return " | ".join(source_bits)

    def _background_job_running(self):
        return bool(self.background_job_process and self.background_job_process.poll() is None)

    def _handle_window_close(self):
        if self._background_job_running():
            try:
                self.background_job_process.terminate()
            except OSError:
                pass
        self.root.destroy()

    def prepare_dataset(self):
        self._start_background_job(
            kind="dataset",
            title="Подготовка датасета",
            code=(
                "from echo_training.pipeline import prepare_or_reuse_dataset; "
                "import sys; "
                "result = prepare_or_reuse_dataset(); "
                "raise SystemExit(0 if result else 1)"
            ),
        )

    def start_training_pipeline(self):
        self._start_background_job(
            kind="training",
            title="Полное обучение",
            code="from echo_training.pipeline import full_main; full_main()",
        )

    def _start_background_job(self, kind, title, code):
        if self._background_job_running():
            active = self.background_job_title or "другая фоновая задача"
            self._append_right(f"[Система] Сначала завершите текущую задачу: {active}.\n\n")
            return

        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        python_path = str(repo_root)
        if env.get("PYTHONPATH"):
            python_path = python_path + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = python_path

        command = [sys.executable, "-u", "-c", code]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            self._append_right(f"[Система] Не удалось запустить задачу '{title}': {exc}\n\n")
            return

        self.background_job_process = process
        self.background_job_kind = kind
        self.background_job_title = title
        self.background_job_started_at = time.time()
        self.background_job_stopping = False
        self.pipeline_status = {
            "active": True,
            "kind": kind,
            "title": title,
            "stage": "Запуск",
            "summary": f"{title}: запуск процесса...",
            "details": "",
            "last_line": "",
            "dataset_rows": self.dataset_snapshot.get("train_data_rows", 0),
            "source_counts": {},
            "checkpoint": self.dataset_snapshot.get("latest_checkpoint", "нет"),
            "loss": "нет",
            "epoch": "нет",
        }
        self._update_background_job_buttons()
        self._append_right(f"[{title}] Процесс запущен. Прогресс будет показан справа и в верхнем блоке.\n\n")
        self.background_job_reader = threading.Thread(
            target=self._read_background_output,
            args=(process, kind, title),
            daemon=True,
        )
        self.background_job_reader.start()

    def _read_background_output(self, process, kind, title):
        try:
            for raw_line in iter(process.stdout.readline, ""):
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                self._append_right_async(f"[{title}] {line}\n")
                self._update_pipeline_from_line(kind, line)
        finally:
            exit_code = process.wait()
            self._finish_background_job(kind, title, exit_code)

    def _extract_number_from_line(self, line):
        match = re.search(r"(\d+)", line)
        if not match:
            return None
        return int(match.group(1))

    def _update_pipeline_from_line(self, kind, line):
        status = self.pipeline_status
        status["last_line"] = line

        if "ШАГ 2: ПОДГОТОВКА ДАННЫХ" in line:
            status["stage"] = "Сбор данных"
            status["summary"] = "Идёт сбор данных для датасета."
        elif "ИСПОЛЬЗУЮ ГОТОВЫЙ ДАТАСЕТ" in line:
            status["stage"] = "Готовый датасет"
            status["summary"] = "Найден готовый датасет, повторная сборка не потребовалась."
        elif "ПЕРЕСБОРКА ДАТАСЕТА" in line:
            status["stage"] = "Пересборка"
            status["summary"] = "Старый датасет признан слабым, выполняется пересборка."
        elif "Загрузка датасетов с Hugging Face" in line:
            status["stage"] = "Внешние источники"
            status["summary"] = "Подключаются внешние датасеты и локальные источники."
        elif "Извлечено локальных диалогов:" in line:
            status["source_counts"]["dialogues"] = self._extract_number_from_line(line) or 0
        elif "Извлечено уроков учителя:" in line:
            status["source_counts"]["teacher_lessons"] = self._extract_number_from_line(line) or 0
        elif "Извлечено знаний:" in line:
            status["source_counts"]["knowledge_base"] = self._extract_number_from_line(line) or 0
        elif "Извлечено из файлов знаний:" in line:
            status["source_counts"]["knowledge_files"] = self._extract_number_from_line(line) or 0
        elif "Извлечено законов:" in line:
            status["source_counts"]["logic_laws"] = self._extract_number_from_line(line) or 0
        elif "Извлечено из Alice:" in line:
            status["source_counts"]["hf_alice"] = self._extract_number_from_line(line) or 0
        elif "Извлечено из YAGPT:" in line:
            status["source_counts"]["hf_yagpt"] = self._extract_number_from_line(line) or 0
        elif "Итоговый размер датасета:" in line:
            count = self._extract_number_from_line(line) or 0
            status["dataset_rows"] = count
            status["stage"] = "Датасет готов"
            status["summary"] = f"Датасет собран: {count} примеров."
        elif "Размер обучающего датасета:" in line:
            status["dataset_rows"] = self._extract_number_from_line(line) or status["dataset_rows"]
        elif "ОБУЧЕНИЕ ЧЕРЕЗ transformers.Trainer" in line:
            status["stage"] = "Инициализация"
            status["summary"] = "Подготовка модели и Trainer."
        elif "Профиль обучения:" in line:
            status["details"] = line
        elif "Загрузка модели" in line:
            status["stage"] = "Загрузка модели"
            status["summary"] = line
        elif "Применение LoRA адаптеров" in line:
            status["stage"] = "LoRA"
            status["summary"] = "Подключаются LoRA-адаптеры."
        elif "Токенизация датасета" in line:
            status["stage"] = "Токенизация"
            status["summary"] = "Токенизация примеров перед обучением."
        elif "НАЧАЛО ОБУЧЕНИЯ" in line:
            status["stage"] = "Обучение"
            status["summary"] = "Обучение запущено."
        elif line.startswith("Шаг "):
            match = re.search(
                r"Шаг\s+(\d+)\s+из\s+([^,]+),\s+эпоха\s+([^,]+),\s+checkpoint:\s+([^,]+),\s+время:\s+(\d+)\s+сек",
                line,
            )
            if match:
                status["stage"] = "Обучение"
                status["epoch"] = match.group(3).strip()
                status["checkpoint"] = match.group(4).strip()
                status["summary"] = (
                    f"Шаг {match.group(1)}/{match.group(2)} | эпоха {match.group(3)} | "
                    f"ckpt {match.group(4)} | {match.group(5)} сек"
                )
        elif line.startswith("Метрики:"):
            loss_match = re.search(r"loss=([0-9.]+)", line)
            if loss_match:
                status["loss"] = loss_match.group(1)
            status["details"] = line
        elif "Сохранение модели" in line:
            status["stage"] = "Сохранение"
            status["summary"] = "Сохраняются адаптер и токенизатор."
        elif "ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО" in line or "ПОЛНЫЙ ЦИКЛ УСПЕШНО ЗАВЕРШЁН" in line:
            status["stage"] = "Готово"
            status["summary"] = "Полный цикл обучения завершён успешно."
        elif "Критическая ошибка" in line or "ОШИБКА" in line or "Ошибка" in line:
            status["stage"] = "Ошибка"
            status["summary"] = line
        elif "Обучение прервано пользователем." in line:
            status["stage"] = "Остановлено"
            status["summary"] = "Обучение остановлено пользователем."

    def _finish_background_job(self, kind, title, exit_code):
        was_stopping = self.background_job_stopping
        self.background_job_process = None
        self.background_job_kind = None
        self.background_job_title = ""
        self.background_job_started_at = None
        self.background_job_stopping = False
        self.dataset_snapshot = self._collect_dataset_snapshot()

        if was_stopping:
            self.pipeline_status["active"] = False
            self.pipeline_status["stage"] = "Остановлено"
            self.pipeline_status["summary"] = f"{title}: процесс остановлен."
            self._append_right_async(f"[{title}] Процесс остановлен.\n\n")
        elif exit_code == 0:
            self.pipeline_status["active"] = False
            if kind == "dataset":
                rows = self.dataset_snapshot.get("train_data_rows", 0)
                self.pipeline_status["stage"] = "Датасет готов"
                self.pipeline_status["summary"] = f"Датасет подготовлен: {rows} примеров."
            else:
                checkpoint = self.dataset_snapshot.get("latest_checkpoint", "нет")
                self.pipeline_status["stage"] = "Готово"
                self.pipeline_status["summary"] = f"Обучение завершено. Последний checkpoint: {checkpoint}."
            self._append_right_async(f"[{title}] Задача завершена успешно.\n\n")
        else:
            self.pipeline_status["active"] = False
            self.pipeline_status["stage"] = "Ошибка"
            last_line = self.pipeline_status.get("last_line", "").strip()
            self.pipeline_status["summary"] = last_line or f"{title}: процесс завершился с кодом {exit_code}."
            self._append_right_async(
                f"[{title}] Процесс завершился с ошибкой. Код выхода: {exit_code}.\n\n"
            )
    def _collect_primary_skills(self):
        if not (SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager")):
            return []

        preferred = {}
        for name, (_, desc, _) in self.assistant.skill_manager.skills.items():
            current = preferred.get(desc)
            if current is None or self._prefer_skill_name(name, current):
                preferred[desc] = name

        skills = [(name, desc) for desc, name in preferred.items()]
        skills.sort(key=lambda item: item[0])
        return skills

    def _prefer_skill_name(self, candidate, current):
        candidate_ascii = all(ord(ch) < 128 for ch in candidate)
        current_ascii = all(ord(ch) < 128 for ch in current)
        if candidate_ascii != current_ascii:
            return not candidate_ascii
        return len(candidate) < len(current)

    def _insert_boot_message(self):
        ethics_on = "включены" if self.assistant.ethics.get("enabled", True) else "отключены"
        skills_status = "активны" if SKILLS_AVAILABLE else "недоступны (нет skills.py)"
        teacher_status = self.assistant.get_teacher_status_summary() if hasattr(self.assistant, "get_teacher_status_summary") else {}
        student_status = self.assistant.get_student_status_summary() if hasattr(self.assistant, "get_student_status_summary") else {}
        dataset_state = (
            f"{self.dataset_snapshot['train_data_rows']} примеров"
            if self.dataset_snapshot.get("train_data_exists")
            else "ещё не собран"
        )
        self._append_left(
            f"[Учитель] Ollama подключена.\n"
            f"[Модель учителя]: {teacher_status.get('teacher_model', 'не подключена')}\n"
            f"[Роль]: отвечает как внешний учитель для правой панели.\n"
            f"[Уроки учителя]: {teacher_status.get('teacher_lessons', 0)}\n"
            f"[База уроков]: {teacher_status.get('teacher_dataset', 'нет')}\n"
            f"[Учиться]: режим автообучения сейчас выключен, кнопку можно включать как тумблер.\n\n"
        )
        self._append_right(
            f"[Ученик] {student_status.get('student_name', 'Эхо')} запущен.\n"
            f"[Режим ученика]: {student_status.get('student_mode', self.assistant.cognitive_mode)}\n"
            f"[Семантический поиск]: {student_status.get('embedding_status', 'отключен')}\n"
            f"[Память ученика]: {student_status.get('memory_db', 'нет')}\n"
            f"[Уроки от учителя]: {student_status.get('teacher_lessons', 0)}\n"
            f"[Основной датасет]: {dataset_state}\n"
            f"[Последний checkpoint]: {self.dataset_snapshot.get('latest_checkpoint', 'нет')}\n"
            f"[Ограничения]: {ethics_on}\n"
            f"[Скиллы]: {skills_status}\n"
            f"[Лог]: {LOG_FILE}\n"
            f"[Подсказка]: пишите один запрос снизу, учитель и ученик ответят отдельно.\n\n"
        )

    def refresh_runtime_status(self):
        current_task = "задач нет"
        progress_text = self.pipeline_status.get("summary", "Прогресс: ожидание")
        if SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager"):
            snapshot = self.assistant.skill_manager.get_runtime_snapshot()
            current_task = snapshot["active_task"] or current_task
            if not self.pipeline_status.get("active"):
                progress_text = snapshot["summary"]
        learning_state = "ВКЛ" if self.learning_mode_enabled else "ВЫКЛ"
        if self.pipeline_status.get("active"):
            current_task = self.pipeline_status.get("title", current_task)
        self.mode_label.config(text=f"Ученик: {self.assistant.cognitive_mode} | Учиться: {learning_state}")
        self.task_label.config(text=f"Активная задача: {current_task}")
        self.progress_label.config(text=progress_text)
        self.dataset_label.config(
            text=(
                f"{self.pipeline_status.get('stage', 'Ожидание')} | "
                f"loss {self.pipeline_status.get('loss', 'нет')} | "
                f"{self._format_dataset_snapshot()}"
            )
        )
        self._update_background_job_buttons()
        self.refresh_weight_panel()
        self.root.after(800, self.refresh_runtime_status)

    def _register_paned_window(self, name, widget, orientation):
        widget.bind("<Configure>", lambda event, pane_name=name: self._schedule_sash_refresh(pane_name), add="+")
        widget.bind("<ButtonRelease-1>", lambda event, pane_name=name: self._capture_sash_ratio(pane_name), add="+")
        widget._codex_orientation = orientation

    def _initialize_responsive_sashes(self):
        for name in tuple(self.sash_ratios.keys()):
            self._apply_sash_ratio(name)

    def _schedule_sash_refresh(self, name):
        if name in self._pending_sash_updates:
            return
        self._pending_sash_updates.add(name)

        def run():
            self._pending_sash_updates.discard(name)
            self._apply_sash_ratio(name)

        self.root.after_idle(run)

    def _capture_sash_ratio(self, name):
        widget = getattr(self, name, None)
        if not widget:
            return
        try:
            x, y = widget.sash_coord(0)
        except tk.TclError:
            return
        orientation = getattr(widget, "_codex_orientation", tk.VERTICAL)
        span = widget.winfo_height() if orientation == tk.VERTICAL else widget.winfo_width()
        if span <= 1:
            return
        value = y if orientation == tk.VERTICAL else x
        self.sash_ratios[name] = max(0.15, min(0.85, value / span))

    def _apply_sash_ratio(self, name):
        widget = getattr(self, name, None)
        if not widget:
            return
        try:
            orientation = getattr(widget, "_codex_orientation", tk.VERTICAL)
            span = widget.winfo_height() if orientation == tk.VERTICAL else widget.winfo_width()
            if span <= 1:
                return
            ratio = self.sash_ratios.get(name, 0.5)
            value = int(span * ratio)
            if orientation == tk.VERTICAL:
                widget.sash_place(0, 1, value)
            else:
                widget.sash_place(0, value, 1)
        except tk.TclError:
            return

    def refresh_weight_panel(self):
        if not self.weight_vars:
            return
        status = self.assistant.get_student_status_summary()
        weights = status.get("weights", {})
        self.updating_weight_controls = True
        try:
            for key, var in self.weight_vars.items():
                value = float(weights.get(key, 0.0))
                var.set(value)
                self.weight_value_labels[key].config(text=f"{value:.2f}")
            self.weight_mode_label.config(text=f"Режим: {status.get('student_mode', 'stable')}")
            self.weight_change_label.config(text=f"Последнее изменение: {status.get('weight_change', 'нет данных')}")
        finally:
            self.updating_weight_controls = False

    def _insert_skill_msg(self, message):
        self._append_right(f"{message}\n")

    def _append_to_area(self, area, text):
        area.insert(tk.END, text)
        area.see(tk.END)

    def _append_left(self, text):
        self._append_to_area(self.left_chat_area, text)

    def _append_right(self, text):
        self._append_to_area(self.right_chat_area, text)

    def _append_left_async(self, text):
        self._queue_ui("left", text)

    def _append_right_async(self, text):
        self._queue_ui("right", text)

    def _clear_both_panels(self):
        self.left_chat_area.delete("1.0", tk.END)
        self.right_chat_area.delete("1.0", tk.END)

    def _queue_ui(self, target, payload):
        self.ui_queue.put((target, payload))

    def _process_ui_queue(self):
        try:
            while True:
                target, payload = self.ui_queue.get_nowait()
                if target == "left":
                    self._append_left(payload)
                elif target == "right":
                    self._append_right(payload)
                elif target == "clear":
                    self._clear_both_panels()
        except queue.Empty:
            pass
        self.root.after(60, self._process_ui_queue)

    def setup_global_shortcuts(self):
        self.root.bind_all("<Control-KeyPress>", self.handle_control_shortcuts, add="+")
        self.root.bind_all("<Control-Insert>", lambda event: self._copy_focused_widget(), add="+")
        self.root.bind_all("<Shift-Insert>", lambda event: self._paste_into_focused_widget(), add="+")

    def handle_control_shortcuts(self, event):
        key = event.keysym.lower()
        if key in ("z", "я"):
            self.interrupt_active_task()
            return "break"
        if key in ("c", "с"):
            return self._copy_focused_widget()
        if key in ("v", "м"):
            return self._paste_into_focused_widget()
        if key in ("x", "ч"):
            return self._cut_focused_widget()
        if key in ("a", "ф"):
            return self._select_all_focused_widget()
        return None

    def _copy_focused_widget(self):
        widget = self.root.focus_get()
        if self._is_text_widget(widget):
            self._do_copy(widget)
            return "break"
        return None

    def _paste_into_focused_widget(self):
        widget = self.root.focus_get()
        if self._is_text_widget(widget) and widget not in (self.left_chat_area, self.right_chat_area):
            self._do_paste(widget)
            return "break"
        return None

    def _cut_focused_widget(self):
        widget = self.root.focus_get()
        if self._is_text_widget(widget) and widget not in (self.left_chat_area, self.right_chat_area):
            self._do_cut(widget)
            return "break"
        return None

    def _select_all_focused_widget(self):
        widget = self.root.focus_get()
        if self._is_text_widget(widget):
            self._do_select_all(widget)
            return "break"
        return None

    def _is_text_widget(self, widget):
        return isinstance(widget, (tk.Text, tk.Entry))

    def setup_context_menu(self, widget, allow_paste=True, allow_cut=True):
        menu = tk.Menu(widget, tearoff=0, bg="#2d2d2d", fg="#ffffff")

        def popup(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        menu.add_command(label="Копировать", command=lambda: self._do_copy(widget))
        if allow_cut:
            menu.add_command(label="Вырезать", command=lambda: self._do_cut(widget))
        if allow_paste:
            menu.add_command(label="Вставить", command=lambda: self._do_paste(widget))
        menu.add_command(label="Выделить всё", command=lambda: self._do_select_all(widget))
        widget.bind("<Button-3>", popup)

    def _do_copy(self, widget):
        try:
            text = widget.selection_get() if isinstance(widget, tk.Entry) else widget.get("sel.first", "sel.last")
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update()
        except tk.TclError:
            pass

    def _do_cut(self, widget):
        try:
            text = widget.selection_get() if isinstance(widget, tk.Entry) else widget.get("sel.first", "sel.last")
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update()
                widget.delete("sel.first", "sel.last")
        except tk.TclError:
            pass

    def _do_paste(self, widget):
        try:
            text = self.root.clipboard_get()
            if text:
                try:
                    widget.delete("sel.first", "sel.last")
                except tk.TclError:
                    pass
                widget.insert("insert", text)
        except tk.TclError:
            pass

    def _do_select_all(self, widget):
        try:
            if isinstance(widget, tk.Entry):
                widget.selection_range(0, tk.END)
                widget.icursor(tk.END)
            else:
                widget.tag_add("sel", "1.0", tk.END)
                widget.mark_set(tk.INSERT, tk.END)
                widget.see(tk.INSERT)
        except tk.TclError:
            pass

    def focus_input(self):
        self.entry.focus_set()

    def clear_chat(self):
        self._clear_both_panels()
        self._append_left("[Система] Левая панель очищена.\n\n")
        self._append_right("[Система] Правая панель очищена.\n\n")

    def queue_command(self, command, send=False):
        self.entry.delete("1.0", tk.END)
        self.entry.insert("1.0", command)
        self.entry.focus_set()
        if send:
            self.send_message()

    def open_skills_window(self):
        if not (SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager")):
            self._append_right("[Система] skills.py недоступен, окно скиллов открыть нельзя.\n\n")
            return

        if self.skills_window and self.skills_window.winfo_exists():
            self.skills_window.focus_force()
            return

        self.skills_window = tk.Toplevel(self.root)
        self.skills_window.title("Скиллы Эхо")
        self.skills_window.geometry("520x520")
        self.skills_window.minsize(420, 360)
        self.skills_window.configure(bg="#0f141a")

        tk.Label(
            self.skills_window,
            text="Быстрый запуск скиллов",
            font=("Segoe UI", 15, "bold"),
            bg="#0f141a",
            fg="#f6f8fb",
        ).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(
            self.skills_window,
            text="Продвинутые команды обучения и сборки. Основные кнопки уже вынесены на главный экран.",
            font=("Segoe UI", 10),
            bg="#0f141a",
            fg="#8aa2b8",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        list_frame = tk.Frame(self.skills_window, bg="#0f141a")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        canvas = tk.Canvas(list_frame, bg="#0f141a", bd=0, highlightthickness=0)
        scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="#0f141a")

        inner.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for skill_name, description in self.primary_skills:
            card = tk.Frame(inner, bg="#16202b", padx=12, pady=10)
            card.pack(fill=tk.X, pady=6)

            tk.Label(
                card,
                text=f"/{skill_name}",
                font=("Consolas", 11, "bold"),
                bg="#16202b",
                fg="#9fd3ff",
            ).pack(anchor="w")
            tk.Label(
                card,
                text=description,
                font=("Segoe UI", 10),
                bg="#16202b",
                fg="#d5dde6",
                wraplength=360,
                justify=tk.LEFT,
            ).pack(anchor="w", pady=(4, 8))

            actions = tk.Frame(card, bg="#16202b")
            actions.pack(fill=tk.X)
            self._create_quick_button(
                actions,
                "Вставить",
                lambda command=f"/{skill_name}": self.queue_command(command, send=False),
            ).pack(side=tk.LEFT)
            self._create_quick_button(
                actions,
                "Запустить",
                lambda command=f"/{skill_name}": self.queue_command(command, send=True),
                accent=True,
            ).pack(side=tk.LEFT, padx=(8, 0))

    def interrupt_active_task(self):
        if self._background_job_running():
            self.background_job_stopping = True
            self.pipeline_status["summary"] = f"{self.background_job_title}: отправлен сигнал остановки..."
            self.pipeline_status["stage"] = "Остановка"
            try:
                self.background_job_process.terminate()
                self._append_right_async(
                    f"[{self.background_job_title}] Отправлен сигнал остановки. Ждём завершения процесса.\n\n"
                )
            except OSError as exc:
                self._append_right_async(
                    f"[{self.background_job_title}] Не удалось остановить процесс: {exc}\n\n"
                )
            return
        if SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager"):
            interrupted, message = self.assistant.skill_manager.interrupt_running_skill()
            self.task_label.config(text=message)
            if interrupted:
                self._append_right_async(f"[Система] {message}\n\n")
            return
        self.task_label.config(text="Прерывание недоступно: skills.py не подключён.")

    def start_file_learning(self):
        self._append_right("[Система] Правая панель обрабатывает файлы знаний...\n")

        def run_async():
            try:
                with self.assistant.db_lock:
                    count = self.assistant.process_knowledge_inbox()
                self._append_right_async(f"[Система] Готово. Усвоено файлов: {count}.\n\n")
            except Exception as e:
                self._append_right_async(f"[Система] Ошибка: {e}\n\n")

        threading.Thread(target=run_async, daemon=True).start()

    def teach_from_left_panel(self):
        self._append_right("[Система] Учитель передаёт последний урок ученику...\n")

        def run_async():
            try:
                with self.assistant.db_lock:
                    success, message = self.assistant.teach_echo_from_ollama()
                if success:
                    self._append_left_async("[Учитель] Последний ответ сохранён как урок для ученика.\n\n")
                    self._append_right_async(f"[Обучение] {message}\n\n")
                else:
                    self._append_right_async(f"[Система] {message}\n\n")
            except Exception as e:
                self._append_right_async(f"[Система] Ошибка обучения: {e}\n\n")

        threading.Thread(target=run_async, daemon=True).start()

    def handle_enter(self, event):
        self.send_message()
        return "break"

    def insert_newline(self, event):
        self.entry.insert(tk.INSERT, "\n")
        return "break"

    def send_message(self, event=None):
        if self.message_in_flight:
            self._append_right("[Система] Дождитесь завершения текущего запроса.\n\n")
            return
        user_text = self.entry.get("1.0", tk.END).strip()
        if not user_text:
            return
        self._append_left(f"Вы: {user_text}\n")
        self._append_right(f"Вы: {user_text}\n")
        self.entry.delete("1.0", tk.END)
        self.focus_input()
        self.message_in_flight = True

        def run_async():
            assistant_name = self.assistant.personality.get("name", "Эхо")
            auto_learning_success = False
            auto_learning_message = ""
            teacher_help_used = False
            try:
                with self.assistant.db_lock:
                    if user_text.startswith("/"):
                        right_response = self.assistant.generate_echo_response(user_text)
                        left_response = "Системные команды выполняются только в правой панели Эхо."
                    else:
                        right_response = self.assistant.generate_echo_response(user_text)
                        needs_teacher_help = bool(
                            self.learning_mode_enabled
                            and getattr(self.assistant, "last_echo_analysis", {}).get("needs_teacher_help")
                        )
                        left_response = self.assistant.generate_ollama_response(user_text)
                        if needs_teacher_help and self.assistant.teacher_answer_usable(left_response):
                            right_response = self.assistant.build_echo_teacher_response(left_response)
                            teacher_help_used = True
                        if self.learning_mode_enabled:
                            auto_learning_success, auto_learning_message = self.assistant.teach_echo_from_ollama()
                self._append_left_async(f"Ollama: {left_response}\n\n")
                self._append_right_async(f"{assistant_name}: {right_response}\n\n")
                if teacher_help_used:
                    self._append_right_async(
                        "[Учитель -> Эхо] Ученик не был уверен и уточнил ответ у Ollama.\n\n"
                    )
                if self.learning_mode_enabled:
                    if auto_learning_success:
                        self._append_left_async("[Учитель] Урок автоматически передан ученику.\n\n")
                        self._append_right_async(f"[Автообучение] {auto_learning_message}\n\n")
                    elif auto_learning_message:
                        self._append_right_async(f"[Автообучение] {auto_learning_message}\n\n")
            except Exception as e:
                logger.error(f"Ошибка ядра: {e}")
                self._append_left_async(f"[Ошибка]: {e}\n\n")
                self._append_right_async(f"[Ошибка]: {e}\n\n")
            finally:
                self.message_in_flight = False

        threading.Thread(target=run_async, daemon=True).start()

    def teach_response(self):
        teach_window = tk.Toplevel(self.root)
        teach_window.title("Ручное подтверждение знания")
        teach_window.geometry("500x300")
        teach_window.configure(bg="#1a1a1a")
        tk.Label(teach_window, text="Контекст:", font=("Arial", 10), bg="#1a1a1a", fg="#ffffff").pack(pady=5)
        user_entry = tk.Entry(teach_window, width=60, font=("Consolas", 10), bg="#252526", fg="#ffffff")
        user_entry.pack(pady=5)
        tk.Label(teach_window, text="Знание:", font=("Arial", 10), bg="#1a1a1a", fg="#ffffff").pack(pady=5)
        answer_entry = tk.Entry(teach_window, width=60, font=("Consolas", 10), bg="#252526", fg="#ffffff")
        answer_entry.pack(pady=5)
        self.setup_context_menu(user_entry, allow_paste=True, allow_cut=True)
        self.setup_context_menu(answer_entry, allow_paste=True, allow_cut=True)

        def save_learning():
            user_text = user_entry.get().strip()
            answer = answer_entry.get().strip()
            if user_text and answer:
                with self.assistant.db_lock:
                    self.assistant.learn(user_text, answer)
                self._append_right(f"[Система] Знание подтверждено: '{answer}'\n\n")
                teach_window.destroy()

        tk.Button(
            teach_window,
            text="Зафиксировать",
            font=("Arial", 10, "bold"),
            bg="#2d2d2d",
            fg="#ffffff",
            command=save_learning,
        ).pack(pady=20)

    def start_proactive_pulse(self):
        def pulse_loop():
            time.sleep(15)
            while True:
                time.sleep(900)
                with self.assistant.db_lock:
                    smart_question = self.assistant.generate_curiosity_question()
                if smart_question:
                    assistant_name = self.assistant.personality.get("name", "Эхо")
                    self._append_right_async(
                        f"{assistant_name}: [{self.assistant.cognitive_mode.upper()}] {smart_question}\n\n"
                    )

        threading.Thread(target=pulse_loop, daemon=True).start()


def main():
    logger.info("=" * 60)
    logger.info("ЗАПУСК ПРИЛОЖЕНИЯ ЭХО")
    logger.info("=" * 60)
    root = tk.Tk()
    AssistantGUI(root)
    root.mainloop()
    logger.info("Приложение закрыто")
