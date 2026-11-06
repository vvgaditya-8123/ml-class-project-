import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure imports from local folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from newspaper_reader import NewspaperReader

class NewspaperReaderGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Newspaper Reader - AI Layout & OCR Pipeline")
        self.root.geometry("1200x820")
        self.root.minsize(1000, 700)

        # Sleek Modern Dark Palette
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

        # Pipeline state
        self.reader = None
        self.current_img_bgr = None
        self.current_img_path = None
        self.current_analysis = None
        self.selected_article_id = None
        self.preview_tk = None

        # Sample images directory
        self.news_dir = os.path.join(BASE_DIR, "layout", "News")
        self.sample_files = []
        if os.path.exists(self.news_dir):
            self.sample_files = [f for f in sorted(os.listdir(self.news_dir)) if f.lower().endswith(('.jpg', '.png'))]

        # Initialize UI & Models
        self._build_ui()
        self.root.after(100, self._init_pipeline)

    def _init_pipeline(self):
        try:
            self.status_bar.config(text="⏳ Initializing AI Models (Layout Model + OCR Model)...")
            self.root.update_idletasks()
            self.reader = NewspaperReader()
            layout_name = type(self.reader.layout_engine).__name__
            ocr_name = getattr(self.reader.ocr_engine, "model_name", "SequenceOCR")
            self.status_bar.config(text=f"⚡ Models Loaded: {layout_name} + {ocr_name} | Ready")

            # Load first sample automatically if available
            if self.sample_files:
                first_sample = os.path.join(self.news_dir, self.sample_files[0])
                self.load_image(first_sample)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to initialize models:\n{str(e)}")
            self.status_bar.config(text=f"❌ Error loading models: {str(e)}")

    def _build_ui(self):
        # Main layout container
        main_box = tk.Frame(self.root, bg=self.c_bg)
        main_box.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # ---------------- HEADER & TOOLBAR ----------------
        header_frame = tk.Frame(main_box, bg=self.c_bg)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title & Subtitle
        title_box = tk.Frame(header_frame, bg=self.c_bg)
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box,
            text="📰 Newspaper Article Reader & Layout Analyzer",
            font=("Segoe UI", 16, "bold"),
            fg=self.c_text,
            bg=self.c_bg
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box,
            text="Dual AI Pipeline: Visual Layout Segmentation + Deep Learning OCR Character/Word Extraction",
            font=("Segoe UI", 9),
            fg=self.c_subtext,
            bg=self.c_bg
        )
        sub_lbl.pack(anchor="w")

        # Controls Toolbar (Browse + Samples dropdown + Run)
        ctrl_frame = tk.Frame(header_frame, bg=self.c_bg)
        ctrl_frame.pack(side=tk.RIGHT)

        browse_btn = tk.Button(
            ctrl_frame,
            text="📁 Browse Photo...",
            font=("Segoe UI", 9, "bold"),
            bg=self.c_card,
            fg=self.c_text,
            activebackground=self.c_border,
            activeforeground=self.c_text,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.browse_file
        )
        browse_btn.pack(side=tk.LEFT, padx=4)

        if self.sample_files:
            sample_lbl = tk.Label(ctrl_frame, text="Samples:", font=("Segoe UI", 9), fg=self.c_subtext, bg=self.c_bg)
            sample_lbl.pack(side=tk.LEFT, padx=(8, 2))

            self.sample_var = tk.StringVar(value=self.sample_files[0])
            sample_menu = ttk.Combobox(
                ctrl_frame,
                textvariable=self.sample_var,
                values=self.sample_files[:30],
                state="readonly",
                width=18
            )
            sample_menu.pack(side=tk.LEFT, padx=2)
            sample_menu.bind("<<ComboboxSelected>>", self._on_sample_selected)

        self.run_btn = tk.Button(
            ctrl_frame,
            text="⚡ Analyze & Read",
            font=("Segoe UI", 9, "bold"),
            bg=self.c_primary,
            fg="#ffffff",
            activebackground=self.c_primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.run_pipeline
        )
        self.run_btn.pack(side=tk.LEFT, padx=6)

        # ---------------- 3-COLUMN WORKSPACE ----------------
        workspace = tk.Frame(main_box, bg=self.c_bg)
        workspace.pack(fill=tk.BOTH, expand=True)

        # 1. Left Panel: Headlines List
        left_col = tk.Frame(workspace, bg=self.c_panel, width=320, highlightthickness=1, highlightbackground=self.c_border)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8))
        left_col.pack_propagate(False)
        self._build_headlines_panel(left_col)

        # 2. Middle Panel: Article Reader
        mid_col = tk.Frame(workspace, bg=self.c_panel, highlightthickness=1, highlightbackground=self.c_border)
        mid_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self._build_reader_panel(mid_col)

        # 3. Right Panel: Newspaper Visualizer
        right_col = tk.Frame(workspace, bg=self.c_panel, width=380, highlightthickness=1, highlightbackground=self.c_border)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_col.pack_propagate(False)
        self._build_visualizer_panel(right_col)

        # ---------------- FOOTER STATUS BAR ----------------
        self.status_bar = tk.Label(
            main_box,
            text="⚡ Status: Ready",
            font=("Segoe UI", 9),
            fg=self.c_subtext,
            bg=self.c_bg,
            anchor="w"
        )
        self.status_bar.pack(fill=tk.X, pady=(6, 0))

    def _build_headlines_panel(self, parent):
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
            text="0 Headlines",
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg=self.c_primary,
            padx=6,
            pady=2
        )
        self.count_badge.pack(side=tk.RIGHT)

        hint = tk.Label(
            parent,
            text="Click any headline below to view its article:",
            font=("Segoe UI", 8),
            fg=self.c_subtext,
            bg=self.c_panel,
            padx=12
        )
        hint.pack(anchor="w", pady=(0, 6))

        # Scrollable headlines listbox
        list_frame = tk.Frame(parent, bg=self.c_panel, padx=8, pady=4)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.headlines_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
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

    def _build_reader_panel(self, parent):
        top_bar = tk.Frame(parent, bg=self.c_panel, padx=14, pady=10)
        top_bar.pack(fill=tk.X)

        title = tk.Label(
            top_bar,
            text="📖 Article Reader",
            font=("Segoe UI", 11, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        title.pack(side=tk.LEFT)

        copy_btn = tk.Button(
            top_bar,
            text="📋 Copy Text",
            font=("Segoe UI", 8, "bold"),
            bg=self.c_card,
            fg=self.c_text,
            activebackground=self.c_border,
            activeforeground=self.c_text,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self.copy_article_text
        )
        copy_btn.pack(side=tk.RIGHT, padx=4)

        save_btn = tk.Button(
            top_bar,
            text="💾 Save Article",
            font=("Segoe UI", 8, "bold"),
            bg=self.c_card,
            fg=self.c_text,
            activebackground=self.c_border,
            activeforeground=self.c_text,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self.save_article_text
        )
        save_btn.pack(side=tk.RIGHT)

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
            font=("Segoe UI", 11),
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

    def _build_visualizer_panel(self, parent):
        header = tk.Frame(parent, bg=self.c_panel, padx=12, pady=10)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="🖼️ Newspaper Layout Map",
            font=("Segoe UI", 11, "bold"),
            fg=self.c_text,
            bg=self.c_panel
        )
        title.pack(side=tk.LEFT)

        key_hint = tk.Label(
            parent,
            text="🟩 Headline | 🟨 Article Body",
            font=("Segoe UI", 8, "bold"),
            fg=self.c_accent,
            bg=self.c_panel,
            padx=12
        )
        key_hint.pack(anchor="w", pady=(0, 4))

        # Canvas for scaled newspaper rendering
        canvas_frame = tk.Frame(parent, bg="#000000", padx=2, pady=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(canvas_frame, bg="#050811", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        f = filedialog.askopenfilename(
            title="Select Newspaper Image Scan",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
        )
        if f:
            self.load_image(f)
            self.run_pipeline()

    def _on_sample_selected(self, event=None):
        selected_name = self.sample_var.get()
        fpath = os.path.join(self.news_dir, selected_name)
        if os.path.exists(fpath):
            self.load_image(fpath)
            self.run_pipeline()

    def load_image(self, fpath: str):
        self.current_img_path = fpath
        self.current_img_bgr = cv2.imread(fpath)
        if self.current_img_bgr is not None:
            self.render_canvas_preview(self.current_img_bgr)
            self.status_bar.config(text=f"Loaded Newspaper: {os.path.basename(fpath)} ({self.current_img_bgr.shape[1]}x{self.current_img_bgr.shape[0]})")

    def run_pipeline(self):
        if self.current_img_bgr is None:
            messagebox.showinfo("No Image", "Please select or browse a newspaper image first.")
            return

        if not self.reader:
            messagebox.showerror("Error", "Models not yet initialized.")
            return

        self.status_bar.config(text="🔍 Analyzing visual layout and running sequence OCR extraction...")
        self.run_btn.config(state=tk.DISABLED)

        # Execute OCR pipeline in worker thread to prevent freezing the UI
        def worker():
            try:
                res = self.reader.process_newspaper(self.current_img_bgr)
                self.root.after(0, self._on_pipeline_success, res)
            except Exception as e:
                self.root.after(0, self._on_pipeline_error, str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_pipeline_success(self, analysis_result: dict):
        self.run_btn.config(state=tk.NORMAL)
        self.current_analysis = analysis_result
        articles = self.current_analysis["articles"]

        # Populate headlines listbox
        self.headlines_listbox.delete(0, tk.END)
        self.count_badge.config(text=f"{len(articles)} Headlines")

        for a in articles:
            hl = a["headline"] if a["headline"] else f"[Region #{a['id']} (Empty OCR)]"
            preview = hl[:45] + ("..." if len(hl) > 45 else "")
            self.headlines_listbox.insert(tk.END, f" #{a['id']}: {preview}")

        if articles:
            self.headlines_listbox.selection_set(0)
            self.display_article(1)
        else:
            self.headline_lbl.config(text="No headlines or articles detected.")
            self.body_text_widget.delete("1.0", tk.END)

        self.status_bar.config(
            text=f"✅ Analysis Complete: Detected {len(articles)} articles with layout segmentation & sequence OCR."
        )

    def _on_pipeline_error(self, err_msg: str):
        self.run_btn.config(state=tk.NORMAL)
        messagebox.showerror("Processing Error", f"An error occurred during analysis:\n{err_msg}")
        self.status_bar.config(text=f"❌ Processing failed: {err_msg}")

    def _on_headline_selected(self, event=None):
        selection = self.headlines_listbox.curselection()
        if selection:
            idx = selection[0]
            article_id = idx + 1
            self.display_article(article_id)

    def display_article(self, article_id: int):
        if not self.current_analysis:
            return

        articles = self.current_analysis["articles"]
        target = next((a for a in articles if a["id"] == article_id), None)
        if not target:
            return

        self.selected_article_id = article_id

        # Update Reader Pane with honest layout vs OCR scores
        hl = target.get("headline", "").strip()
        display_hl = hl if hl else f"[Region #{article_id} - Headline Text Empty]"
        self.headline_lbl.config(text=display_hl)

        layout_score_str = f"{target['layout_score']*100:.1f}%" if target.get('layout_score') is not None else "N/A"
        ocr_score_str = f"{target['headline_ocr_score']*100:.1f}%" if target.get('headline_ocr_score') is not None else "N/A"

        meta_line = (
            f"Article #{article_id}  |  "
            f"Layout Score: {layout_score_str}  |  "
            f"OCR Score: {ocr_score_str}  |  "
            f"Words: {target['word_count']}  |  "
            f"Columns: {len(target['body_boxes'])}"
        )
        self.article_meta_lbl.config(text=meta_line)

        self.body_text_widget.delete("1.0", tk.END)
        body = target.get("body_text", "").strip()
        if body:
            self.body_text_widget.insert("1.0", body)
        else:
            # Honest status message: no fake strings
            empty_note = (
                "[No body text recognized for this article layout region.]\n\n"
                "• Visual bounding boxes remain visible on the newspaper preview map.\n"
                "• The OCR recognizer returned empty text for the column blocks."
            )
            self.body_text_widget.insert("1.0", empty_note)

        # Update Visual Overlay on the Newspaper Map
        overlay = self.reader.render_visual_overlay(self.current_img_bgr, articles, selected_id=article_id)
        self.render_canvas_preview(overlay)

    def render_canvas_preview(self, img_bgr: np.ndarray):
        """Scales image proportionally to fit visualizer canvas."""
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()

        if c_w <= 1 or c_h <= 1:
            c_w, c_h = 360, 520

        h, w = img_bgr.shape[:2]
        scale = min(c_w / float(w), c_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)
        self.preview_tk = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        x_pos = (c_w - new_w) // 2
        y_pos = (c_h - new_h) // 2
        self.canvas.create_image(x_pos, y_pos, anchor="nw", image=self.preview_tk)

    def copy_article_text(self):
        txt = self.body_text_widget.get("1.0", tk.END).strip()
        hl = self.headline_lbl.cget("text")
        full_clip = f"{hl}\n\n{txt}"
        if full_clip:
            self.root.clipboard_clear()
            self.root.clipboard_append(full_clip)
            messagebox.showinfo("Copied", "Headline and article text copied to clipboard!")

    def save_article_text(self):
        txt = self.body_text_widget.get("1.0", tk.END).strip()
        hl = self.headline_lbl.cget("text")
        if not txt:
            return
        f = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if f:
            with open(f, "w", encoding="utf-8") as out:
                out.write(f"{hl}\n{'='*len(hl)}\n\n{txt}\n")
            messagebox.showinfo("Saved", f"Article saved to {os.path.basename(f)}!")

def main():
    root = tk.Tk()
    app = NewspaperReaderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

