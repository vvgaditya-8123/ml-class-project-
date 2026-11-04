import os
import sys

# Ensure local .venv packages can be resolved even if run with default global python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
venv_site = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

import tkinter as tk
from tkinter import ttk, messagebox
import time

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Import our inference engine
from predict import EmojiPredictor

class EmojiDetectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Emoji Predictor - Real-Time Keystroke Intelligence")
        self.root.geometry("980x740")
        self.root.minsize(850, 650)

        # Color Palette - Sleek Modern Dark/Midnight Theme
        self.c_bg = "#0f172a"          # slate-900
        self.c_panel = "#1e293b"       # slate-800
        self.c_card = "#334155"        # slate-700
        self.c_text = "#f8fafc"        # slate-50
        self.c_subtext = "#94a3b8"     # slate-400
        self.c_primary = "#6366f1"     # indigo-500
        self.c_primary_hover = "#4f46e5"
        self.c_accent = "#10b981"      # emerald-500
        self.c_border = "#475569"

        self.root.configure(bg=self.c_bg)

        # Typography configuration
        self.font_hero_emoji = ("Segoe UI Emoji", 56)
        self.font_title = ("Segoe UI", 18, "bold")
        self.font_sub = ("Segoe UI", 10)
        self.font_body = ("Segoe UI", 11)
        self.font_btn = ("Segoe UI", 10, "bold")
        self.font_emoji_btn = ("Segoe UI Emoji", 20)
        self.font_stat = ("Segoe UI", 9)

        # State & debounce timer
        self.debounce_timer = None
        self.predictor = None

        # Try to load predictor
        self._load_predictor()

        # Build UI layout
        self._build_ui()

        # Initial prediction trigger
        self.trigger_prediction()

    def _load_predictor(self):
        try:
            self.predictor = EmojiPredictor()
        except Exception as e:
            self.predictor = None
            messagebox.showerror(
                "Model Not Ready",
                f"Could not load emoji model:\n{str(e)}\n\nPlease ensure train.py has completed."
            )

    def _build_ui(self):
        # Main Container
        main_container = tk.Frame(self.root, bg=self.c_bg)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # ---------------- HEADER ----------------
        header_frame = tk.Frame(main_container, bg=self.c_bg)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_lbl = tk.Label(
            header_frame,
            text="✨ Real-Time Emoji Detection & Prediction",
            font=self.font_title,
            fg=self.c_text,
            bg=self.c_bg
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header_frame,
            text="Type naturally in the box below to see intelligent emoji predictions update dynamically with each keystroke.",
            font=self.font_sub,
            fg=self.c_subtext,
            bg=self.c_bg
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # ---------------- CONTENT SPLIT (Two columns) ----------------
        content_frame = tk.Frame(main_container, bg=self.c_bg)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Input Box & Controls
        left_col = tk.Frame(content_frame, bg=self.c_panel, padx=16, pady=16, highlightthickness=1, highlightbackground=self.c_border)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right Column: Prediction Results & Top-K display
        right_col = tk.Frame(content_frame, bg=self.c_panel, padx=16, pady=16, highlightthickness=1, highlightbackground=self.c_border)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._build_left_column(left_col)
        self._build_right_column(right_col)

        # ---------------- FOOTER / STATUS BAR ----------------
        self.status_bar = tk.Label(
            main_container,
            text="⚡ Status: Ready | Latency: 0.0 ms | Model: Multi-gram TF-IDF + Logistic Regression",
            font=self.font_stat,
            fg=self.c_subtext,
            bg=self.c_bg,
            anchor="w"
        )
        self.status_bar.pack(fill=tk.X, pady=(10, 0))

    def _build_left_column(self, parent):
        # Section Label
        lbl = tk.Label(
            parent,
            text="📝 Your Text Input",
            font=("Segoe UI", 12, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        lbl.pack(anchor="w", pady=(0, 8))

        # Text input box
        self.text_input = tk.Text(
            parent,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#ffffff",
            selectbackground=self.c_primary,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=12,
            height=8
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)

        # Bind keystrokes for real-time predictions
        self.text_input.bind("<KeyRelease>", self._on_key_release)

        # Default sample text
        default_prompt = "Merry Christmas and happy holidays with family! "
        self.text_input.insert("1.0", default_prompt)

        # Text Stats & Action buttons row
        action_row = tk.Frame(parent, bg=self.c_panel)
        action_row.pack(fill=tk.X, pady=(10, 10))

        self.char_lbl = tk.Label(
            action_row,
            text=f"Chars: {len(default_prompt)} | Words: {len(default_prompt.split())}",
            font=self.font_stat,
            fg=self.c_subtext,
            bg=self.c_panel
        )
        self.char_lbl.pack(side=tk.LEFT)

        clear_btn = tk.Button(
            action_row,
            text="Clear",
            font=self.font_btn,
            bg=self.c_card,
            fg=self.c_text,
            activebackground=self.c_border,
            activeforeground=self.c_text,
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.clear_text
        )
        clear_btn.pack(side=tk.RIGHT, padx=(6, 0))

        copy_btn = tk.Button(
            action_row,
            text="Copy Text",
            font=self.font_btn,
            bg=self.c_primary,
            fg="#ffffff",
            activebackground=self.c_primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.copy_text
        )
        copy_btn.pack(side=tk.RIGHT)

        # Quick preset examples label
        presets_lbl = tk.Label(
            parent,
            text="💡 Quick Test Prompts (Click to try):",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        presets_lbl.pack(anchor="w", pady=(8, 6))

        # Preset buttons container
        presets_frame = tk.Frame(parent, bg=self.c_panel)
        presets_frame.pack(fill=tk.X)

        samples = [
            ("🎄 Christmas Party", "Merry Christmas and happy holidays to all!"),
            ("🔥 Absolute Fire", "This song and beat is absolute fire right now"),
            ("😂 Hilarious Joke", "LMAO that was the funniest thing I have ever heard"),
            ("☀️ Sunny Beach", "Enjoying the warm bright sun at the beach today"),
            ("❤️ Love & Affection", "I love you so much with all my heart sweetheart"),
            ("🇺🇸 4th of July", "Happy 4th of July parade celebration America"),
            ("📸 Photo Shoot", "Took stunning pictures with my new camera lens"),
            ("💯 100% Real", "Keeping it 100 percent real all day every day")
        ]

        # Arrange preset buttons in 2 columns
        for idx, (title, text_val) in enumerate(samples):
            btn = tk.Button(
                presets_frame,
                text=title,
                font=("Segoe UI", 9),
                bg=self.c_card,
                fg=self.c_text,
                activebackground=self.c_border,
                activeforeground=self.c_text,
                relief=tk.FLAT,
                padx=8,
                pady=4,
                cursor="hand2",
                command=lambda t=text_val: self.load_preset(t)
            )
            row = idx // 2
            col = idx % 2
            btn.grid(row=row, column=col, sticky="ew", padx=3, pady=3)

        presets_frame.columnconfigure(0, weight=1)
        presets_frame.columnconfigure(1, weight=1)

    def _build_right_column(self, parent):
        # Section Label
        lbl = tk.Label(
            parent,
            text="🎯 Top Prediction",
            font=("Segoe UI", 12, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        lbl.pack(anchor="w", pady=(0, 8))

        # Hero Prediction Card
        self.hero_card = tk.Frame(parent, bg=self.c_card, padx=16, pady=16, highlightthickness=1, highlightbackground=self.c_border)
        self.hero_card.pack(fill=tk.X, pady=(0, 16))

        # Big Hero Emoji
        self.hero_emoji_lbl = tk.Label(
            self.hero_card,
            text="🎄",
            font=self.font_hero_emoji,
            fg="#ffffff",
            bg=self.c_card
        )
        self.hero_emoji_lbl.pack()

        # Hero Emoji Name & Confidence
        self.hero_name_lbl = tk.Label(
            self.hero_card,
            text="Christmas Tree",
            font=("Segoe UI", 13, "bold"),
            fg=self.c_text,
            bg=self.c_card
        )
        self.hero_name_lbl.pack(pady=(4, 2))

        self.hero_conf_lbl = tk.Label(
            self.hero_card,
            text="Confidence: 94.2%",
            font=self.font_sub,
            fg=self.c_accent,
            bg=self.c_card
        )
        self.hero_conf_lbl.pack()

        # Insert Hero Emoji Button
        self.insert_hero_btn = tk.Button(
            self.hero_card,
            text="Insert Emoji Into Text ↵",
            font=self.font_btn,
            bg=self.c_primary,
            fg="#ffffff",
            activebackground=self.c_primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
            command=self.insert_top_emoji
        )
        self.insert_hero_btn.pack(pady=(10, 0))

        # Suggestions Label
        sug_header = tk.Label(
            parent,
            text="📊 Top 5 Suggested Emojis (Click any to insert):",
            font=("Segoe UI", 11, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        sug_header.pack(anchor="w", pady=(6, 6))

        # Container for Top 5 suggestion rows
        self.sug_container = tk.Frame(parent, bg=self.c_panel)
        self.sug_container.pack(fill=tk.BOTH, expand=True)

        self.suggestion_widgets = []
        for i in range(5):
            row_frame = tk.Frame(self.sug_container, bg=self.c_card, padx=8, pady=5)
            row_frame.pack(fill=tk.X, pady=3)

            # Clickable Emoji Button
            emoji_btn = tk.Button(
                row_frame,
                text="🎄",
                font=self.font_emoji_btn,
                bg=self.c_card,
                fg="#ffffff",
                activebackground=self.c_border,
                activeforeground="#ffffff",
                relief=tk.FLAT,
                width=3,
                cursor="hand2"
            )
            emoji_btn.pack(side=tk.LEFT, padx=(0, 8))

            # Info Frame: Name + Progress Bar
            info_frame = tk.Frame(row_frame, bg=self.c_card)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            text_row = tk.Frame(info_frame, bg=self.c_card)
            text_row.pack(fill=tk.X)

            name_lbl = tk.Label(
                text_row,
                text="Christmas Tree",
                font=("Segoe UI", 10, "bold"),
                fg=self.c_text,
                bg=self.c_card,
                anchor="w"
            )
            name_lbl.pack(side=tk.LEFT)

            pct_lbl = tk.Label(
                text_row,
                text="92.1%",
                font=("Segoe UI", 9),
                fg=self.c_subtext,
                bg=self.c_card,
                anchor="e"
            )
            pct_lbl.pack(side=tk.RIGHT)

            # Custom progress canvas
            bar_canvas = tk.Canvas(
                info_frame,
                height=6,
                bg="#1e293b",
                highlightthickness=0
            )
            bar_canvas.pack(fill=tk.X, pady=(4, 0))

            self.suggestion_widgets.append({
                "frame": row_frame,
                "btn": emoji_btn,
                "name": name_lbl,
                "pct": pct_lbl,
                "canvas": bar_canvas,
                "emoji_val": "🎄"
            })

    def _on_key_release(self, event=None):
        """Debounces user keystrokes so typing remains completely smooth without stutter."""
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)
        self.debounce_timer = self.root.after(25, self.trigger_prediction)

    def trigger_prediction(self):
        """Fetches current text, runs inference, and updates the display."""
        text = self.text_input.get("1.0", tk.END).strip()

        # Update stats
        chars = len(text)
        words = len(text.split()) if text else 0
        self.char_lbl.config(text=f"Chars: {chars} | Words: {words}")

        if not self.predictor:
            return

        # Perform fast ML inference
        result = self.predictor.predict(text, top_k=5)
        suggestions = result["suggestions"]
        latency = result["latency_ms"]

        # Update Status Bar
        self.status_bar.config(
            text=f"⚡ Status: Real-time Active | Keystroke Latency: {latency:.2f} ms | Classes: 20 Emojis"
        )

        if not suggestions:
            self.hero_emoji_lbl.config(text="❓")
            self.hero_name_lbl.config(text="Type something...")
            self.hero_conf_lbl.config(text="Confidence: 0.0%")
            for w in self.suggestion_widgets:
                w["btn"].config(text="", state=tk.DISABLED)
                w["name"].config(text="")
                w["pct"].config(text="")
                w["canvas"].delete("all")
            return

        # Update Hero Card
        top = suggestions[0]
        self.hero_emoji_lbl.config(text=top["emoji"])
        self.hero_name_lbl.config(text=top["name"])
        self.hero_conf_lbl.config(text=f"Confidence: {top['percentage']}")
        self.insert_hero_btn.config(command=lambda e=top["emoji"]: self.insert_emoji(e))

        # Update Top 5 List
        for idx, item in enumerate(suggestions):
            if idx >= len(self.suggestion_widgets):
                break
            w = self.suggestion_widgets[idx]
            emoji_char = item["emoji"]
            score = item["score"]

            w["btn"].config(
                text=emoji_char,
                state=tk.NORMAL,
                command=lambda e=emoji_char: self.insert_emoji(e)
            )
            w["name"].config(text=item["name"])
            w["pct"].config(text=item["percentage"])
            w["emoji_val"] = emoji_char

            # Draw progress bar
            canvas = w["canvas"]
            canvas.delete("all")
            w_width = canvas.winfo_width()
            if w_width <= 1:
                w_width = 200  # Default fallback width before initial layout render
            bar_len = int(w_width * max(0.02, score))

            # Color highlight: Top 1 is emerald/primary, others are indigo/cyan
            bar_color = self.c_accent if idx == 0 else self.c_primary
            canvas.create_rectangle(0, 0, bar_len, 6, fill=bar_color, width=0)

    def insert_emoji(self, emoji_char: str):
        """Inserts selected emoji at current text cursor position."""
        if not emoji_char:
            return
        self.text_input.insert(tk.INSERT, f" {emoji_char} ")
        self.text_input.focus_set()
        self.trigger_prediction()

    def insert_top_emoji(self):
        current_hero = self.hero_emoji_lbl.cget("text")
        if current_hero and current_hero != "❓":
            self.insert_emoji(current_hero)

    def load_preset(self, text: str):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", text)
        self.text_input.focus_set()
        self.trigger_prediction()

    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.text_input.focus_set()
        self.trigger_prediction()

    def copy_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Text copied to clipboard!")

def launch_gui():
    root = tk.Tk()
    app = EmojiDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()

