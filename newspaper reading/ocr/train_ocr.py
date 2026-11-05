import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data2", "training_data")
TEST_DIR = os.path.join(BASE_DIR, "data2", "testing_data")
MODEL_OUT = os.path.join(BASE_DIR, "char_ocr_model.pth")
CLASSES_OUT = os.path.join(BASE_DIR, "char_classes.json")

# 36 alphanumeric classes
CLASSES = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class CharDataset(Dataset):
    def __init__(self, root_dir, classes, img_size=(32, 32)):
        self.samples = []
        self.img_size = img_size
        
        for c in classes:
            c_dir = os.path.join(root_dir, c)
            if not os.path.exists(c_dir):
                continue
            idx = CLASS_TO_IDX[c]
            for fname in os.listdir(c_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(c_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        # Read grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.img_size, dtype=np.uint8)
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

        # Invert so character strokes are positive / white (1.0) and background is 0.0
        # If mean > 127, it's black text on light background
        if np.mean(img) > 127:
            img = 255 - img

        # Normalize to [-1.0, 1.0]
        tensor = torch.from_numpy(img).float() / 127.5 - 1.0
        tensor = tensor.unsqueeze(0) # (1, H, W)
        return tensor, label

class CharCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 (32x32 -> 16x16)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 -> 64 (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 -> 128 (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(128 * 4 * 4, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train():
    print("=" * 60)
    print("  OCR CHARACTER RECOGNITION MODEL TRAINING (36 CLASSES)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load datasets
    print(f"Loading training data from {TRAIN_DIR}...")
    train_ds = CharDataset(TRAIN_DIR, CLASSES)
    print(f"Loaded {len(train_ds):,} training character images.")

    test_ds = CharDataset(TEST_DIR, CLASSES)
    print(f"Loaded {len(test_ds):,} testing character images.")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = CharCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)

    epochs = 9
    print(f"\nStarting training for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / total

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                test_correct += (preds == batch_y).sum().item()
                test_total += len(batch_y)

        test_acc = test_correct / test_total if test_total > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    print(f"\nTraining completed in {time.time()-t0:.2f}s.")

    # Save model and classes
    print(f"Saving model to {MODEL_OUT}...")
    torch.save(model.state_dict(), MODEL_OUT)
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)
    print(f"Model saved ({os.path.getsize(MODEL_OUT)/(1024*1024):.2f} MB).")
    print(f"Classes saved to {CLASSES_OUT}.")

if __name__ == "__main__":
    train()

