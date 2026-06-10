import threading
import time
import tkinter as tk
from tkinter import scrolledtext

from .core import LOG_FILE, SKILLS_AVAILABLE, SYSTEM_VERSION, UnifiedAssistant, logger


class AssistantGUI:
    def __init__(self, root):
        self.assistant = UnifiedAssistant()
        self.root = root
        self.skills_window = None
        self.primary_skills = self._collect_primary_skills()

        assistant_name = self.assistant.personality.get("name", "Эхо")
        self.root.title(f"{SYSTEM_VERSION} — {assistant_name}")
        self.root.geometry("1180x760")
        self.root.minsize(860, 560)
        self.root.configure(bg="#0f141a")

        self._build_layout()
        self.setup_global_shortcuts()

        if SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager"):
            def _append_skill_log(message):
                self.chat_area.after(0, lambda m=message: self._insert_skill_msg(m))

            self.assistant.skill_manager.register_gui(_append_skill_log)

            def _clear_chat():
                self.chat_area.after(0, lambda: self.chat_area.delete("1.0", tk.END))

            self.assistant._gui_clear_callback = _clear_chat

        self._insert_boot_message()
        self.refresh_runtime_status()
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
        self._build_chat_panel()
        self._build_composer()

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
            ("💬", "Фокус", self.focus_input, False),
            ("✨", "Скиллы", self.open_skills_window, False),
            ("🚀", "Учиться", lambda: self.queue_command("/учиться", send=True), False),
            ("🧪", "Тест", lambda: self.queue_command("/тест", send=True), False),
            ("🛠", "Сборка", lambda: self.queue_command("/собрать", send=True), False),
            ("📚", "Знания", self.start_file_learning, False),
            ("🧠", "Ручное", self.teach_response, False),
            ("📊", "Стат", lambda: self.queue_command("/стат", send=True), False),
            ("🧹", "Чисто", self.clear_chat, False),
            ("■", "Стоп", self.interrupt_active_task, True),
        ]

        for icon, label, command, danger in buttons:
            self._create_sidebar_button(icon, label, command, danger=danger)

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
            text="Диалоговая консоль Эхо",
            font=("Segoe UI", 18, "bold"),
            bg="#0f141a",
            fg="#f5f7fa",
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Чат, память, обучение и скиллы в одном окне.",
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

    def _build_chat_panel(self):
        chat_outer = tk.Frame(self.workspace, bg="#26313c", bd=0, padx=1, pady=1)
        chat_outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 12))

        self.chat_area = scrolledtext.ScrolledText(
            chat_outer,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#101821",
            fg="#d7e0e9",
            insertbackground="#ffffff",
            bd=0,
            highlightthickness=0,
            padx=14,
            pady=14,
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)

        self.setup_context_menu(self.chat_area, allow_paste=False, allow_cut=False)

    def _build_composer(self):
        composer = tk.Frame(self.workspace, bg="#0f141a")
        composer.pack(fill=tk.X, padx=18, pady=(0, 16))

        quick_actions = tk.Frame(composer, bg="#0f141a")
        quick_actions.pack(fill=tk.X, pady=(0, 8))

        self._create_quick_button(quick_actions, "Учиться", lambda: self.queue_command("/учиться", send=True)).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Тест", lambda: self.queue_command("/тест", send=True)).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Сборка", lambda: self.queue_command("/собрать", send=True)).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Скиллы", self.open_skills_window).pack(side=tk.LEFT, padx=(0, 8))
        self._create_quick_button(quick_actions, "Помощь", lambda: self.queue_command("/помощь", send=True)).pack(side=tk.LEFT, padx=(0, 8))
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
        status_text = "АКТИВЕН (нейросетевой поиск)" if self.assistant.embedding_model else "ОТКЛЮЧЕН (поиск по словам)"
        ethics_on = "включены" if self.assistant.ethics.get("enabled", True) else "отключены"
        skills_status = "активны" if SKILLS_AVAILABLE else "недоступны (нет skills.py)"
        self._append_chat(
            f"[Ядро] {SYSTEM_VERSION} запущено.\n"
            f"[Статус ИИ]: {status_text}\n"
            f"[Режим]: {self.assistant.cognitive_mode}\n"
            f"[Ограничения]: {ethics_on}\n"
            f"[Скиллы]: {skills_status}\n"
            f"[Лог]: {LOG_FILE}\n"
            f"[Подсказка]: Левая панель для быстрых действий, кнопка «Скиллы» открывает команды.\n\n"
        )

    def refresh_runtime_status(self):
        current_task = "задач нет"
        progress_text = "Статус: ожидание"
        if SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager"):
            snapshot = self.assistant.skill_manager.get_runtime_snapshot()
            current_task = snapshot["active_task"] or current_task
            progress_text = snapshot["summary"]
        self.mode_label.config(text=f"Режим: {self.assistant.cognitive_mode}")
        self.task_label.config(text=f"Активная задача: {current_task}")
        self.progress_label.config(text=progress_text)
        self.root.after(800, self.refresh_runtime_status)

    def _insert_skill_msg(self, message):
        self._append_chat(f"{message}\n")

    def _append_chat(self, text):
        self.chat_area.insert(tk.END, text)
        self.chat_area.see(tk.END)

    def _append_chat_async(self, text):
        self.chat_area.after(0, lambda t=text: self._append_chat(t))

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
        if self._is_text_widget(widget) and widget is not self.chat_area:
            self._do_paste(widget)
            return "break"
        return None

    def _cut_focused_widget(self):
        widget = self.root.focus_get()
        if self._is_text_widget(widget) and widget is not self.chat_area:
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
        self.chat_area.delete("1.0", tk.END)
        self._append_chat("[Система] Чат очищен.\n\n")

    def queue_command(self, command, send=False):
        self.entry.delete("1.0", tk.END)
        self.entry.insert("1.0", command)
        self.entry.focus_set()
        if send:
            self.send_message()

    def open_skills_window(self):
        if not (SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager")):
            self._append_chat("[Система] skills.py недоступен, окно скиллов открыть нельзя.\n\n")
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
            text="Нажми кнопку, чтобы вставить или сразу выполнить команду.",
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
        if SKILLS_AVAILABLE and hasattr(self.assistant, "skill_manager"):
            interrupted, message = self.assistant.skill_manager.interrupt_running_skill()
            self.task_label.config(text=message)
            if interrupted:
                self._append_chat_async(f"[Система] {message}\n\n")
            return
        self.task_label.config(text="Прерывание недоступно: skills.py не подключён.")

    def start_file_learning(self):
        self._append_chat("[Система] Обработка файлов...\n")

        def run_async():
            try:
                with self.assistant.db_lock:
                    count = self.assistant.process_knowledge_inbox()
                self._append_chat_async(f"[Система] Готово. Усвоено: {count}.\n\n")
            except Exception as e:
                self._append_chat_async(f"[Система] Ошибка: {e}\n\n")

        threading.Thread(target=run_async, daemon=True).start()

    def handle_enter(self, event):
        self.send_message()
        return "break"

    def insert_newline(self, event):
        self.entry.insert(tk.INSERT, "\n")
        return "break"

    def send_message(self, event=None):
        user_text = self.entry.get("1.0", tk.END).strip()
        if not user_text:
            return
        self._append_chat(f"Вы: {user_text}\n")
        self.entry.delete("1.0", tk.END)
        assistant_name = self.assistant.personality.get("name", "Эхо")
        try:
            with self.assistant.db_lock:
                response = self.assistant.generate_response(user_text)
            self._append_chat(f"{assistant_name}: {response}\n\n")
        except Exception as e:
            logger.error(f"Ошибка ядра: {e}")
            self._append_chat(f"[Ошибка]: {e}\n\n")
        self.focus_input()

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
                self._append_chat(f"[Система] Знание подтверждено: '{answer}'\n\n")
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
                    self._append_chat_async(
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
