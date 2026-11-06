import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import psutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.crnn_recognizer import CRNNRecognizer, VOCAB_LIST, CHAR_TO_IDX, BLANK_IDX
from datasets.base_loader import OCRItem
from preprocessing.augmentor import NewspaperAugmentor

@dataclass
class TrainingConfig:
    max_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    freeze_backbone: bool = False
    checkpoint_dir: str = os.path.join(BASE_DIR, "checkpoints", "finetuned_recognition")
    patience: int = 5
    pretrained_weights_path: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    target_img_height: int = 32
    max_img_width: int = 640
    weight_decay: float = 1e-4

class NewspaperTextLineDataset(Dataset):
    """
    Dataset loader for fine-tuning text-line recognition with CTC targets.
    Preserves exact character case and aspect ratios.
    """
    def __init__(self, items: List[OCRItem], img_height: int = 32, max_width: int = 640, augment: bool = False):
        self.items = items
        self.img_height = img_height
        self.max_width = max_width
        self.augmentor = NewspaperAugmentor() if augment else None

    def __len__(self):
        return len(self.items)

    def text_to_labels(self, text: str) -> List[int]:
        labels = []
        unknown_chars = []
        for ch in text:
            if ch in CHAR_TO_IDX:
                labels.append(CHAR_TO_IDX[ch])
            else:
                unknown_chars.append(ch)

        if unknown_chars:
            # Report vocabulary mismatch without silently deleting
            print(f"Warning: Character(s) not in vocabulary: {set(unknown_chars)} in text '{text}'")

        return labels if labels else [CHAR_TO_IDX.get(" ", 1)]

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(item.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.img_height, 64), 255, dtype=np.uint8)

        # Apply crop if bbox given
        if item.bbox is not None:
            bx, by, bw, bh = item.bbox
            ih, iw = img.shape
            if bw > 5 and bh > 5 and bx < iw and by < ih:
                img = img[max(0, by):min(ih, by+bh), max(0, bx):min(iw, bx+bw)]

        # Augmentation
        if self.augmentor:
            img = self.augmentor.augment(img)

        # Invert so black text becomes white strokes on dark background
        if np.mean(img) > 127:
            img = 255 - img

        # Resize preserving aspect ratio to height 32
        h, w = img.shape[:2]
        scale = float(self.img_height) / float(max(1, h))
        new_w = max(16, int(w * scale))
        new_w = min(self.max_width, new_w)
        resized = cv2.resize(img, (new_w, self.img_height), interpolation=cv2.INTER_AREA)

        # Pad to max_width preserving aspect ratio
        canvas = np.zeros((self.img_height, self.max_width), dtype=np.float32)
        canvas[:, :new_w] = (resized.astype(np.float32) / 127.5) - 1.0

        tensor = torch.from_numpy(canvas).unsqueeze(0)  # (1, 32, W)
        labels = self.text_to_labels(item.transcription)

        # Compute output sequence length after CNN subsampling (4x horizontal pooling)
        valid_seq_len = max(1, (new_w // 4) - 1)

        return tensor, torch.tensor(labels, dtype=torch.long), valid_seq_len, len(labels)

def ctc_collate_fn(batch):
    images, labels_list, seq_lens, target_lens = zip(*batch)
    images = torch.stack(images, 0)
    flat_targets = torch.cat(labels_list)
    input_lengths = torch.tensor(seq_lens, dtype=torch.long)
    target_lengths = torch.tensor(target_lens, dtype=torch.long)
    return images, flat_targets, input_lengths, target_lengths

class FineTunePipeline:
    """
    Custom transfer learning & fine-tuning pipeline for newspaper OCR:
    - Backbone layer freezing/unfreezing with safety verification
    - Connectionist Temporal Classification (CTC) Loss
    - Learning rate scheduling with Cosine Annealing
    - Checkpoint recovery (resuming interrupted jobs)
    - Early stopping
    - RAM / VRAM telemetry
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg = config or TrainingConfig()
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CRNNRecognizer(img_height=self.cfg.target_img_height).to(self.device)
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

        if self.cfg.freeze_backbone:
            if not self.cfg.pretrained_weights_path or not os.path.exists(self.cfg.pretrained_weights_path):
                raise ValueError(
                    "Cannot freeze a randomly initialized backbone. "
                    "Provide compatible pretrained weights or train the full model."
                )
            state = torch.load(self.cfg.pretrained_weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
            print("Pretrained weights loaded successfully; freezing CNN backbone.")
            self.model.freeze_backbone()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.max_epochs
        )

        self.start_epoch = 1
        self.best_val_loss = float("inf")

        if self.cfg.resume_checkpoint and os.path.exists(self.cfg.resume_checkpoint):
            self.resume(self.cfg.resume_checkpoint)

    def log_resource_usage(self):
        ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        gpu_info = "N/A"
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_info = f"{gpu_mem:.1f} MB VRAM"
        return f"System RAM: {ram_gb:.2f} GB | GPU: {gpu_info}"

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for imgs, targets, input_lens, target_lens in dataloader:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)  # (B, W, Classes)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (W, B, Classes) for CTCLoss

            loss = self.criterion(log_probs, targets, input_lens, target_lens)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for imgs, targets, input_lens, target_lens in dataloader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(imgs)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)

                loss = self.criterion(log_probs, targets, input_lens, target_lens)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

    def fit(self, train_items: List[OCRItem], val_items: List[OCRItem]):
        train_ds = NewspaperTextLineDataset(train_items, self.cfg.target_img_height, self.cfg.max_img_width, augment=True)
        val_ds = NewspaperTextLineDataset(val_items, self.cfg.target_img_height, self.cfg.max_img_width, augment=False)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=ctc_collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=ctc_collate_fn, num_workers=0)

        print(f"Beginning fine-tuning ({len(train_items)} train, {len(val_items)} val samples)...")
        patience_counter = 0

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step()

            duration = time.time() - t0
            telemetry = self.log_resource_usage()
            print(f"[Epoch {epoch:02d}/{self.cfg.max_epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {duration:.1f}s | {telemetry}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(self.cfg.checkpoint_dir, "best_model.pth")
                self.save(ckpt_path, epoch, val_loss)
                print(f"  ★ New best model checkpoint saved -> {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def save(self, path: str, epoch: int, val_loss: float):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "vocab": VOCAB_LIST
        }, path)

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed training from epoch {self.start_epoch} (Best Val Loss: {self.best_val_loss:.4f})")
