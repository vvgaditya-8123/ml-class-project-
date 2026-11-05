import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.dataset_manager import DatasetManager
from datasets.base_loader import OCRItem
from training.finetune_pipeline import FineTunePipeline, TrainingConfig

def main():
    parser = argparse.ArgumentParser(description="Newspaper OCR Fine-Tuning Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="", help="Optional dataset directory")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume")
    args = parser.parse_args()

    dm = DatasetManager()
    
    # Check default Standard OCR dataset location
    standard_ocr_path = os.path.join(os.path.dirname(BASE_DIR), "ocr", "data")
    items = []
    if os.path.exists(standard_ocr_path):
        print(f"Loading paired samples from {standard_ocr_path}...")
        raw_items = dm.load_dataset("standard_ocr", standard_ocr_path, split_name="training_data")
        # Sample for demonstration fine-tuning
        items = raw_items[:2000]

    if not items:
        print("No external dataset path specified; generating paired training crops from newspaper repository...")
        # Create synthetic test pairs from layout News images
        news_dir = os.path.join(os.path.dirname(BASE_DIR), "layout", "News")
        if os.path.exists(news_dir):
            for f in os.listdir(news_dir)[:5]:
                items.append(OCRItem(
                    image_path=os.path.join(news_dir, f),
                    transcription="NEWSPAPER HEADLINE COVERAGE",
                    bbox=(20, 100, 450, 40),
                    dataset_source="Local News"
                ))

    splits = dm.build_splits(items, train_ratio=0.85, val_ratio=0.15, test_ratio=0.0)
    cfg = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_checkpoint=args.resume if args.resume else None
    )

    pipeline = FineTunePipeline(cfg)
    pipeline.fit(splits.train, splits.val)

if __name__ == "__main__":
    main()
