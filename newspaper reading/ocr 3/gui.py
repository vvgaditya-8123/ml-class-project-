import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure Paddle environment variables are set before imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(BASE_DIR, ".paddlex"))
os.environ.setdefault("PADDLE_HOME", os.path.join(BASE_DIR, ".paddle"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ocr3_pipeline import RobustNewspaperOCRPipeline
from pdf_handler import NewspaperDocumentHandler

# Optional TTS support
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


class NewspaperOCR3GUI:
    """
    State-of-the-art GUI for OCR 3 AI Newspaper Reader:
    - Multi-engine recognition: PaddleOCR PP-OCRv4, PP-OCRv3, and Custom Fine-Tuned PyTorch CTC Model.
    - PDF & Image document handling (multi-page support via pypdfium2).
    - Intelligent layout segmentation: groups multi-column text under its governing headline.
    - Interactive 3-panel workspace: Headline selector, article text reader, and dynamic newspaper layout visualizer.
    - Click-to-select on visualizer canvas and copy/export/TTS utilities.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Newspaper Reader - OCR 3 Architecture")
        self.root.geometry("1280x860")
        self.root.minsize(1050, 720)

        # Elegant Modern Slate Theme
        self.c_bg = "#0b0f19"          # deep midnight slate
        self.c_panel = "#151e2e"       # slate panel
        self.c_card = "#1f293d"        # slate card
        self.c_border = "#334155"      # border slate
        self.c_text = "#f8fafc"        # primary white text
        self.c_subtext = "#94a3b8"     # muted slate text
        self.c_primary = "#3b82f6"     # vibrant blue
        self.c_primary_hover = "#2563eb"
        self.c_accent = "#10b981"      # emerald green
        self.c_gold = "#f59e0b"        # amber gold

        self.root.configure(bg=self.c_bg)

        # Pipeline & Document State
        self.pipeline = None
        self.current_doc_path = None
        self.current_pages = []          # List of NumPy BGR images
        self.current_page_idx = 0        # 0-indexed active page
        self.current_analysis = None
        self.current_articles = []
        self.selected_article_id = None
        self.preview_tk = None
        self.scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.displayed_w = 0
        self.displayed_h = 0
        self.reader_font_size = 11
        self.is_processing = False

        # Preloaded samples from layout/News or ocr 3
        self.sample_files = []
        self._find_sample_files()

        # Build UI layout
        self._build_ui()

        # Asynchronous model pipeline initialization
        self.root.after(80, self._async_init_pipeline)

    def _find_sample_files(self):
        """Locates sample newspaper scans."""
        candidates = [
            os.path.join(os.path.dirname(BASE_DIR), "layout", "News"),
            os.path.join(BASE_DIR, "samples"),
            os.path.join(BASE_DIR, "data")
        ]
        for c in candidates:
            if os.path.exists(c):
                files = [os.path.join(c, f) for f in sorted(os.listdir(c)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.pdf'))]
                if files:
                    self.sample_files.extend(files)

    def _async_init_pipeline(self):
        """Initializes the OCR 3 pipeline in background thread."""
        self.status_bar.config(text="⏳ Initializing AI Models (PaddleOCR Detector + Fine-Tuned Recognition Engine)...")

        def worker():
            try:
                # Default to the custom fine-tuned model if available, else PP-OCRv4
                finetuned_path = os.path.join(BASE_DIR, "checkpoints", "finetuned_recognition", "best_finetuned_model.pt")
                default_model = "custom_finetuned" if os.path.exists(finetuned_path) else "en_PP-OCRv4"

                pipeline = RobustNewspaperOCRPipeline(use_angle_cls=True, recognition_model=default_model)
                self.root.after(0, lambda: self._on_pipeline_ready(pipeline, default_model))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: self._on_pipeline_failed(err_msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_pipeline_ready(self, pipeline, active_model: str):
        self.pipeline = pipeline
        model_label = "Custom Fine-Tuned PyTorch CTC" if active_model == "custom_finetuned" else "PaddleOCR PP-OCRv4"
        self.model_combo.set(model_label)
        self.status_bar.config(
            text=f"⚡ Ready | Model: {model_label} | Preprocessor & Multi-Column Layout Active"
        )
        # Auto-load the first available sample
        if self.sample_files:
            self.load_document(self.sample_files[0], auto_run=True)

    def _on_pipeline_failed(self, error: str):
        messagebox.showerror("Model Initialization Error", f"Failed to initialize OCR pipeline:\n{error}")
        self.status_bar.config(text=f"❌ Error loading models: {error}")

    def _build_ui(self):
        main_box = tk.Frame(self.root, bg=self.c_bg)
        main_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # ---------------- TOP HEADER & CONTROLS ----------------
        header_frame = tk.Frame(main_box, bg=self.c_bg)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title & Subtitle
        title_box = tk.Frame(header_frame, bg=self.c_bg)
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box,
            text="📰 AI Newspaper Reader & Layout Analyzer (OCR 3)",
            font=("Segoe UI", 15, "bold"),
            fg=self.c_text,
            bg=self.c_bg
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box,
            text="Pretrained PaddleOCR Detection • Custom Fine-Tuned CTC Recognition • Multi-Column Layout Grouping",
            font=("Segoe UI", 9),
            fg=self.c_subtext,
            bg=self.c_bg
        )
        sub_lbl.pack(anchor="w")

        # Controls Toolbar
        ctrl_frame = tk.Frame(header_frame, bg=self.c_bg)
        ctrl_frame.pack(side=tk.RIGHT)

        # Model Selector Dropdown
        model_lbl = tk.Label(ctrl_frame, text="Model:", font=("Segoe UI", 9, "bold"), fg=self.c_subtext, bg=self.c_bg)
        model_lbl.pack(side=tk.LEFT, padx=(6, 2))

        self.model_var = tk.StringVar(value="Custom Fine-Tuned PyTorch CTC")
        self.model_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.model_var,
            values=[
                "Custom Fine-Tuned PyTorch CTC",
                "PaddleOCR PP-OCRv4 (Pretrained)",
                "PaddleOCR PP-OCRv3 (Lightweight)"
            ],
            state="readonly",
            width=26
        )
        self.model_combo.pack(side=tk.LEFT, padx=3)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        # Samples Dropdown
        if self.sample_files:
            s_lbl = tk.Label(ctrl_frame, text="Samples:", font=("Segoe UI", 9), fg=self.c_subtext, bg=self.c_bg)
            s_lbl.pack(side=tk.LEFT, padx=(8, 2))

            sample_names = [os.path.basename(p) for p in self.sample_files[:30]]
            self.sample_var = tk.StringVar(value=sample_names[0])
            self.sample_combo = ttk.Combobox(
                ctrl_frame,
                textvariable=self.sample_var,
                values=sample_names,
                state="readonly",
                width=16
            )
            self.sample_combo.pack(side=tk.LEFT, padx=3)
            self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

        # Browse Button
        browse_btn = tk.Button(
            ctrl_frame,
            text="📁 Open File...",
            font=("Segoe UI", 9, "bold"),
            bg=self.c_card,
            fg=self.c_text,
            activebackground=self.c_border,
            activeforeground=self.c_text,
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2",
            command=self.browse_file
        )
        browse_btn.pack(side=tk.LEFT, padx=4)

        # Run Analysis Button
        self.run_btn = tk.Button(
            ctrl_frame,
            text="⚡ Analyze & Read",
            font=("Segoe UI", 9, "bold"),
            bg=self.c_primary,
            fg="#ffffff",
            activebackground=self.c_primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=13,
            pady=5,
            cursor="hand2",
            command=self.run_pipeline_thread
        )
        self.run_btn.pack(side=tk.LEFT, padx=6)

        # ---------------- 3-COLUMN WORKSPACE ----------------
        workspace = tk.Frame(main_box, bg=self.c_bg)
        workspace.pack(fill=tk.BOTH, expand=True)

        # 1. Left Panel: Detected Headlines
        left_col = tk.Frame(workspace, bg=self.c_panel, width=320, highlightthickness=1, highlightbackground=self.c_border)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8))
        left_col.pack_propagate(False)
        self._build_headlines_panel(left_col)

        # 2. Middle Panel: Article Reader
        mid_col = tk.Frame(workspace, bg=self.c_panel, highlightthickness=1, highlightbackground=self.c_border)
        mid_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self._build_reader_panel(mid_col)

        # 3. Right Panel: Layout Map & Visualizer
        right_col = tk.Frame(workspace, bg=self.c_panel, width=420, highlightthickness=1, highlightbackground=self.c_border)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_col.pack_propagate(False)
        self._build_visualizer_panel(right_col)

        # ---------------- FOOTER STATUS BAR ----------------
        footer_frame = tk.Frame(main_box, bg=self.c_bg)
        footer_frame.pack(fill=tk.X, pady=(6, 0))

        self.status_bar = tk.Label(
            footer_frame,
            text="⚡ Status: Initializing...",
            font=("Segoe UI", 9),
            fg=self.c_subtext,
            bg=self.c_bg,
            anchor="w"
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_bar = ttk.Progressbar(footer_frame, mode="indeterminate", length=120)
        # Packed dynamically when running

    def _build_headlines_panel(self, parent: tk.Frame):
        header = tk.Frame(parent, bg=self.c_panel, padx=12, pady=10)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="📑 Detected Headlines",
            font=("Segoe UI", 11, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        title.pack(side=tk.LEFT)

        self.count_badge = tk.Label(
            header,
            text="0 Articles",
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg=self.c_primary,
            padx=7,
            pady=2
        )
        self.count_badge.pack(side=tk.RIGHT)

        hint = tk.Label(
            parent,
            text="Click a headline below to reveal its article text:",
            font=("Segoe UI", 8),
            fg=self.c_subtext,
            bg=self.c_panel,
            padx=12
        )
        hint.pack(anchor="w", pady=(0, 4))

        # Search / filter box
        search_box = tk.Frame(parent, bg=self.c_panel, padx=10, pady=4)
        search_box.pack(fill=tk.X)

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_box,
            textvariable=self.search_var,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.c_border
        )
        search_entry.pack(fill=tk.X)

        # Headlines listbox container
        list_frame = tk.Frame(parent, bg=self.c_panel, padx=8, pady=6)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.headlines_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#f8fafc",
            selectbackground=self.c_primary,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
            activestyle="none"
        )
        self.headlines_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.headlines_listbox.yview)

        self.headlines_listbox.bind("<<ListboxSelect>>", self._on_headline_selected)

        # Connect search filter trace after listbox is instantiated
        self._set_placeholder(search_entry, "🔍 Filter headlines...")
        self.search_var.trace_add("write", self._on_filter_changed)

        # PDF Page Selector Bar (bottom of left column)
        self.page_nav_frame = tk.Frame(parent, bg=self.c_card, padx=6, pady=6)
        self.page_nav_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.prev_page_btn = tk.Button(
            self.page_nav_frame,
            text="◀ Prev",
            font=("Segoe UI", 8, "bold"),
            bg=self.c_panel,
            fg=self.c_text,
            relief=tk.FLAT,
            padx=6,
            pady=2,
            command=self._prev_page
        )
        self.prev_page_btn.pack(side=tk.LEFT)

        self.page_lbl = tk.Label(
            self.page_nav_frame,
            text="Page 1 / 1",
            font=("Segoe UI", 8, "bold"),
            fg=self.c_subtext,
            bg=self.c_card
        )
        self.page_lbl.pack(side=tk.LEFT, expand=True)

        self.next_page_btn = tk.Button(
            self.page_nav_frame,
            text="Next ▶",
            font=("Segoe UI", 8, "bold"),
            bg=self.c_panel,
            fg=self.c_text,
            relief=tk.FLAT,
            padx=6,
            pady=2,
            command=self._next_page
        )
        self.next_page_btn.pack(side=tk.RIGHT)

    def _build_reader_panel(self, parent: tk.Frame):
        top_bar = tk.Frame(parent, bg=self.c_panel, padx=12, pady=8)
        top_bar.pack(fill=tk.X)

        title = tk.Label(
            top_bar,
            text="📖 Article Reader",
            font=("Segoe UI", 11, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        title.pack(side=tk.LEFT)

        # Font size adjustment buttons
        font_box = tk.Frame(top_bar, bg=self.c_panel)
        font_box.pack(side=tk.RIGHT, padx=(8, 0))

        zoom_out_btn = tk.Button(
            font_box, text="A-", font=("Segoe UI", 8, "bold"),
            bg=self.c_card, fg=self.c_text, relief=tk.FLAT, padx=6, pady=2,
            command=self._decrease_font
        )
        zoom_out_btn.pack(side=tk.LEFT, padx=1)

        zoom_in_btn = tk.Button(
            font_box, text="A+", font=("Segoe UI", 8, "bold"),
            bg=self.c_card, fg=self.c_text, relief=tk.FLAT, padx=6, pady=2,
            command=self._increase_font
        )
        zoom_in_btn.pack(side=tk.LEFT, padx=1)

        # Export & Copy buttons
        save_btn = tk.Button(
            top_bar,
            text="💾 Save Article",
            font=("Segoe UI", 8, "bold"),
            bg=self.c_card,
            fg=self.c_text,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self.save_article_text
        )
        save_btn.pack(side=tk.RIGHT, padx=4)

        copy_btn = tk.Button(
            top_bar,
            text="📋 Copy Text",
            font=("Segoe UI", 8, "bold"),
            bg=self.c_card,
            fg=self.c_text,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self.copy_article_text
        )
        copy_btn.pack(side=tk.RIGHT, padx=4)

        if HAS_TTS:
            tts_btn = tk.Button(
                top_bar,
                text="🔊 Read Aloud",
                font=("Segoe UI", 8, "bold"),
                bg=self.c_card,
                fg=self.c_accent,
                relief=tk.FLAT,
                padx=8,
                pady=3,
                cursor="hand2",
                command=self.read_aloud_tts
            )
            tts_btn.pack(side=tk.RIGHT, padx=4)

        # Headline Display Header Card
        self.reader_hero = tk.Frame(parent, bg=self.c_card, padx=14, pady=10, highlightthickness=1, highlightbackground=self.c_border)
        self.reader_hero.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.headline_lbl = tk.Label(
            self.reader_hero,
            text="No Headline Selected",
            font=("Segoe UI", 13, "bold"),
            fg="#ffffff",
            bg=self.c_card,
            wraplength=480,
            justify=tk.LEFT
        )
        self.headline_lbl.pack(anchor="w")

        self.article_meta_lbl = tk.Label(
            self.reader_hero,
            text="Select a headline from the left list to read its article.",
            font=("Segoe UI", 9),
            fg=self.c_subtext,
            bg=self.c_card
        )
        self.article_meta_lbl.pack(anchor="w", pady=(4, 0))

        # Article Body Text Area
        body_frame = tk.Frame(parent, bg=self.c_panel, padx=12)
        body_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        body_scroll = tk.Scrollbar(body_frame)
        body_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.body_text_widget = tk.Text(
            body_frame,
            wrap=tk.WORD,
            font=("Segoe UI", self.reader_font_size),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#ffffff",
            selectbackground=self.c_primary,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=14,
            yscrollcommand=body_scroll.set
        )
        self.body_text_widget.pack(fill=tk.BOTH, expand=True)
        body_scroll.config(command=self.body_text_widget.yview)

    def _build_visualizer_panel(self, parent: tk.Frame):
        header = tk.Frame(parent, bg=self.c_panel, padx=12, pady=8)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="🗺️ Visual Layout Map",
            font=("Segoe UI", 11, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        title.pack(side=tk.LEFT)

        key_hint = tk.Label(
            header,
            text="🟩 Headline  🟨 Body",
            font=("Segoe UI", 8, "bold"),
            fg=self.c_accent,
            bg=self.c_panel
        )
        key_hint.pack(side=tk.RIGHT)

        hint = tk.Label(
            parent,
            text="Click on any region in the scan below to select its article:",
            font=("Segoe UI", 8),
            fg=self.c_subtext,
            bg=self.c_panel,
            padx=12
        )
        hint.pack(anchor="w", pady=(0, 4))

        # Canvas for scaled newspaper rendering
        canvas_frame = tk.Frame(parent, bg="#050811", padx=2, pady=2, highlightthickness=1, highlightbackground=self.c_border)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(canvas_frame, bg="#050811", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    # ---------------- EVENT HANDLERS & LOGIC ----------------

    def _set_placeholder(self, entry, text):
        entry.insert(0, text)
        entry.config(fg="#64748b")
        def on_focus_in(event):
            if entry.get() == text:
                entry.delete(0, tk.END)
                entry.config(fg="#f8fafc")
        def on_focus_out(event):
            if not entry.get().strip():
                entry.insert(0, text)
                entry.config(fg="#64748b")
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    def _on_filter_changed(self, *args):
        if not hasattr(self, "headlines_listbox") or self.headlines_listbox is None:
            return
        query = self.search_var.get().strip().lower()
        if query in ("🔍 filter headlines...", ""):
            self._populate_headlines_list(self.current_articles)
            return

        filtered = [a for a in self.current_articles if query in a["headline"].lower() or query in a["body_text"].lower()]
        self._populate_headlines_list(filtered)

    def _on_model_changed(self, event=None):
        selection = self.model_var.get()
        if not self.pipeline:
            return

        finetuned_path = os.path.join(BASE_DIR, "checkpoints", "finetuned_recognition", "best_finetuned_model.pt")
        if "Fine-Tuned" in selection:
            target_model = "custom_finetuned"
            ckpt = finetuned_path if os.path.exists(finetuned_path) else None
        elif "PP-OCRv3" in selection:
            target_model = "en_PP-OCRv3"
            ckpt = None
        else:
            target_model = "en_PP-OCRv4"
            ckpt = None

        self.pipeline.set_recognition_model(target_model, checkpoint_path=ckpt)
        self.status_bar.config(text=f"⚡ Model switched to {selection}. Re-run analysis to apply.")

    def _on_sample_selected(self, event=None):
        selected_name = self.sample_var.get()
        matched = next((p for p in self.sample_files if os.path.basename(p) == selected_name), None)
        if matched and os.path.exists(matched):
            self.load_document(matched, auto_run=True)

    def browse_file(self):
        f = filedialog.askopenfilename(
            title="Select Newspaper Scan or PDF",
            filetypes=[
                ("Newspaper Documents", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.pdf"),
                ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"),
                ("PDF Documents", "*.pdf")
            ]
        )
        if f:
            self.load_document(f, auto_run=True)

    def load_document(self, file_path: str, auto_run: bool = False):
        self.current_doc_path = file_path
        handler = NewspaperDocumentHandler()
        try:
            pages = handler.load_document_pages(file_path)
            if not pages:
                raise ValueError("No readable pages found.")
            self.current_pages = pages
            self.current_page_idx = 0
            self._update_page_nav()
            self._render_current_page_preview()

            doc_name = os.path.basename(file_path)
            self.status_bar.config(
                text=f"Loaded '{doc_name}' ({len(pages)} page(s) | {pages[0].shape[1]}x{pages[0].shape[0]} px)"
            )

            if auto_run and self.pipeline:
                self.run_pipeline_thread()
        except Exception as e:
            messagebox.showerror("Document Load Error", f"Failed to load document:\n{e}")

    def _update_page_nav(self):
        total = max(1, len(self.current_pages))
        curr = self.current_page_idx + 1
        self.page_lbl.config(text=f"Page {curr} / {total}")
        self.prev_page_btn.config(state=tk.NORMAL if self.current_page_idx > 0 else tk.DISABLED)
        self.next_page_btn.config(state=tk.NORMAL if curr < total else tk.DISABLED)

    def _prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self._update_page_nav()
            self.run_pipeline_thread()

    def _next_page(self):
        if self.current_page_idx + 1 < len(self.current_pages):
            self.current_page_idx += 1
            self._update_page_nav()
            self.run_pipeline_thread()

    def run_pipeline_thread(self):
        if self.is_processing:
            return
        if not self.current_pages:
            messagebox.showinfo("No Document", "Please open a newspaper scan or PDF document first.")
            return
        if not self.pipeline:
            messagebox.showinfo("Please Wait", "AI Models are still initializing. Please wait a moment.")
            return

        self.is_processing = True
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.start(10)
        self.status_bar.config(text="🔍 Running Layout Detection, Bounding Box Extraction & OCR Recognition...")

        active_page = self.current_pages[self.current_page_idx]

        def worker():
            try:
                analysis = self.pipeline.process_document(active_page)
                self.root.after(0, lambda: self._on_pipeline_complete(analysis))
            except Exception as e:
                err_text = str(e)
                self.root.after(0, lambda: self._on_pipeline_error(err_text))

        threading.Thread(target=worker, daemon=True).start()

    def _on_pipeline_complete(self, analysis: dict):
        self.is_processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.run_btn.config(state=tk.NORMAL, text="⚡ Analyze & Read")

        self.current_analysis = analysis
        self.current_articles = analysis.get("articles", [])

        self._populate_headlines_list(self.current_articles)

        dur = analysis.get("execution_time_sec", 0.0)

        # Print all detected articles, headlines, and text directly to the console terminal
        print("\n" + "=" * 75)
        print(f"📰 OCR 3 NEWSPAPER ANALYSIS COMPLETE ({dur}s)")
        print(f"   Detected {len(self.current_articles)} Structured Article(s)")
        print("=" * 75)
        for art in self.current_articles:
            clean_hl = " ".join(art.get("headline", "").split())
            print(f"\n[Article #{art['id']}] HEADLINE:")
            print(f"  {clean_hl}")
            print(f"  Confidence: {art.get('confidence', 0.95)*100:.1f}% | Words: {art.get('word_count', 0)} | Columns: {len(art.get('body_boxes', []))}")
            print("  EXTRACTED BODY TEXT:")
            body_txt = art.get("body_text", "").strip()
            if body_txt:
                for line in body_txt.splitlines():
                    print(f"    {line}")
            else:
                print("    (No body text)")
        print("\n" + "=" * 75 + "\n")
        sys.stdout.flush()

        if self.current_articles:
            self.headlines_listbox.selection_set(0)
            self.display_article(self.current_articles[0]["id"])
        else:
            self.headline_lbl.config(text="No headlines detected on this page.")
            self.article_meta_lbl.config(text="Try another page or image with clearer text contrast.")
            self.body_text_widget.delete("1.0", tk.END)
            self._render_current_page_preview()

        self.status_bar.config(
            text=f"✅ Analysis Complete ({dur}s) | Detected {len(self.current_articles)} structured articles with linked body text."
        )

    def _on_pipeline_error(self, err_text: str):
        self.is_processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.run_btn.config(state=tk.NORMAL, text="⚡ Analyze & Read")
        messagebox.showerror("Analysis Error", f"An error occurred during OCR pipeline execution:\n{err_text}")
        self.status_bar.config(text=f"❌ Analysis failed: {err_text}")

    def _populate_headlines_list(self, articles: list):
        self.headlines_listbox.delete(0, tk.END)
        self.count_badge.config(text=f"{len(articles)} Articles")

        for a in articles:
            a_id = a.get("id", a.get("article_id", 1))
            raw_hl = a.get("headline", "").strip() or f"Headline #{a_id}"
            clean_hl = " ".join(raw_hl.split())
            words = a.get("word_count", 0)
            snippet = clean_hl[:38] + ("..." if len(clean_hl) > 38 else "")
            self.headlines_listbox.insert(tk.END, f" #{a_id} {snippet} ({words}w)")

    def _on_headline_selected(self, event=None):
        selection = self.headlines_listbox.curselection()
        if selection and self.current_articles:
            idx = selection[0]
            if idx < len(self.current_articles):
                art = self.current_articles[idx]
                self.display_article(art["id"])

    def display_article(self, article_id: int):
        target = next((a for a in self.current_articles if a.get("id") == article_id or a.get("article_id") == article_id), None)
        if not target:
            return

        self.selected_article_id = article_id

        # Update Reader Pane
        clean_hl = " ".join(target["headline"].split())
        self.headline_lbl.config(text=clean_hl)
        conf_pct = target.get("confidence", 0.95) * 100.0
        words = target.get("word_count", 0)
        cols = len(target.get("body_boxes", []))
        self.article_meta_lbl.config(
            text=f"Article #{article_id}  |  Confidence: {conf_pct:.1f}%  |  Words: {words}  |  Columns/Blocks: {cols}"
        )

        self.body_text_widget.delete("1.0", tk.END)
        self.body_text_widget.insert("1.0", target["body_text"])

        # Print selected article to console
        print(f"\n>>> [Article #{article_id} Selected in UI]")
        print(f"HEADLINE: {clean_hl}")
        print(f"TEXT:\n{target['body_text']}\n")
        sys.stdout.flush()

        # Update Visual Overlay
        self._render_current_page_preview(selected_id=article_id)

    def _render_current_page_preview(self, selected_id: int = None):
        if not self.current_pages:
            return

        page_bgr = self.current_pages[self.current_page_idx]

        if self.pipeline and self.current_articles:
            rendered_bgr = self.pipeline.render_visual_overlay(page_bgr, self.current_articles, selected_id=selected_id)
        else:
            rendered_bgr = page_bgr

        self._draw_image_on_canvas(rendered_bgr)

    def _draw_image_on_canvas(self, img_bgr: np.ndarray):
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        if c_w <= 1 or c_h <= 1:
            c_w, c_h = 400, 560

        h, w = img_bgr.shape[:2]
        scale = min(c_w / float(w), c_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        self.scale_factor = scale
        self.displayed_w = new_w
        self.displayed_h = new_h
        self.img_offset_x = (c_w - new_w) // 2
        self.img_offset_y = (c_h - new_h) // 2

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)
        self.preview_tk = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor="nw", image=self.preview_tk)

    def _on_canvas_resize(self, event=None):
        if self.current_pages:
            self._render_current_page_preview(selected_id=self.selected_article_id)

    def _on_canvas_click(self, event):
        """Detects which headline or article bounding box was clicked on the map."""
        if not self.current_articles or self.scale_factor <= 0:
            return

        # Translate canvas coordinates back to original image space
        orig_x = int((event.x - self.img_offset_x) / self.scale_factor)
        orig_y = int((event.y - self.img_offset_y) / self.scale_factor)

        # Check for intersection with headline boxes first, then body boxes
        clicked_art = None
        for art in self.current_articles:
            hx, hy, hw, hh = art["headline_box"]
            if (hx <= orig_x <= hx + hw) and (hy <= orig_y <= hy + hh):
                clicked_art = art
                break

        if not clicked_art:
            for art in self.current_articles:
                for bx, by, bw, bh in art.get("body_boxes", []):
                    if (bx <= orig_x <= bx + bw) and (by <= orig_y <= by + bh):
                        clicked_art = art
                        break
                if clicked_art:
                    break

        if clicked_art:
            a_id = clicked_art["id"]
            # Synchronize listbox selection
            for i, a in enumerate(self.current_articles):
                if a["id"] == a_id:
                    self.headlines_listbox.selection_clear(0, tk.END)
                    self.headlines_listbox.selection_set(i)
                    self.headlines_listbox.see(i)
                    break
            self.display_article(a_id)

    def _increase_font(self):
        self.reader_font_size = min(22, self.reader_font_size + 1)
        self.body_text_widget.config(font=("Segoe UI", self.reader_font_size))

    def _decrease_font(self):
        self.reader_font_size = max(8, self.reader_font_size - 1)
        self.body_text_widget.config(font=("Segoe UI", self.reader_font_size))

    def copy_article_text(self):
        txt = self.body_text_widget.get("1.0", tk.END).strip()
        hl = self.headline_lbl.cget("text")
        full_clip = f"{hl}\n{'='*len(hl)}\n\n{txt}"
        if txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(full_clip)
            messagebox.showinfo("Copied", "Headline and article text copied to clipboard!")

    def save_article_text(self):
        txt = self.body_text_widget.get("1.0", tk.END).strip()
        hl = self.headline_lbl.cget("text")
        if not txt:
            messagebox.showinfo("Empty Article", "No article text available to save.")
            return

        f = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files (*.txt)", "*.txt"), ("Markdown files (*.md)", "*.md")]
        )
        if f:
            with open(f, "w", encoding="utf-8") as out:
                out.write(f"# {hl}\n\n{txt}\n")
            messagebox.showinfo("Article Saved", f"Saved successfully to:\n{os.path.basename(f)}")

    def read_aloud_tts(self):
        if not HAS_TTS:
            messagebox.showinfo("TTS Notice", "pyttsx3 is not installed for text-to-speech.")
            return

        hl = self.headline_lbl.cget("text")
        body = self.body_text_widget.get("1.0", tk.END).strip()
        speech_text = f"Headline. {hl}. Article body. {body}"

        def tts_worker():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                engine.say(speech_text[:500]) # Speak first 500 characters
                engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")

        threading.Thread(target=tts_worker, daemon=True).start()


def main():
    root = tk.Tk()
    app = NewspaperOCR3GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
