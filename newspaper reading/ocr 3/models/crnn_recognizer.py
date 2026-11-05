import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple

# Standard English vocabulary (numbers, upper, lower, punctuation)
CHAR_VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
BLANK_IDX = 0
VOCAB_LIST = ["<blank>"] + list(CHAR_VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_LIST)}

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden_features, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(hidden_features * 2, out_features)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        return output

class CRNNRecognizer(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN + CTC) for Newspaper Text-Line Recognition.
    Architecture:
      1. CNN Feature Extractor (VGG/ResNet style)
      2. Bidirectional LSTM Sequence Modeler
      3. Linear Projection to Character Logits
      4. CTC (Connectionist Temporal Classification) Decoder
    """

    def __init__(self, img_height: int = 32, num_classes: int = len(VOCAB_LIST)):
        super().__init__()
        self.img_height = img_height
        self.num_classes = num_classes

        # 1. Feature Extractor (CNN)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32xW -> 16x(W/2)

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x(W/2) -> 8x(W/4)

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x(W/4) -> 4x(W/4)

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Block 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x(W/4) -> 2x(W/4)

            # Block 7 (Final compression)
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)  # 2x(W/4) -> 1x(W/4 - 1)
        )

        # 2. Sequence Modeler (BiLSTM)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, H=32, W)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"Expected height=1 after CNN, got {h}"

        conv = conv.squeeze(2)  # (B, C, W)
        conv = conv.permute(0, 2, 1)  # (B, W, C) -> Sequence length is W

        output = self.rnn(conv)  # (B, W, NumClasses)
        return output

    def freeze_backbone(self):
        """Freezes CNN feature extractor for transfer learning on top recurrent layers."""
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    @staticmethod
    def decode_ctc(logits: torch.Tensor) -> Tuple[str, float]:
        """
        Greedy CTC decoder: collapses repeating characters and removes blank tokens (index 0).
        """
        probs = torch.softmax(logits, dim=-1)
        best_indices = torch.argmax(probs, dim=-1).cpu().numpy()
        best_probs = torch.max(probs, dim=-1).values.cpu().numpy()

        decoded_chars = []
        confidences = []
        prev_idx = -1

        for idx, prob in zip(best_indices, best_probs):
            if idx != prev_idx and idx != BLANK_IDX:
                if idx < len(VOCAB_LIST):
                    decoded_chars.append(VOCAB_LIST[idx])
                    confidences.append(float(prob))
            prev_idx = idx

        text = "".join(decoded_chars)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return text, avg_conf
