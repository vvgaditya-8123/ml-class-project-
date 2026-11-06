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
    parser = argparse.ArgumentParser(description="Fine-tune sequence OCR on real newspaper text-line datasets")
    parser.add_argument("--dataset_type", type=str, default="standard_ocr", choices=["standard_ocr", "impact", "prima", "enp"], help="Dataset format")
    parser.add_argument("--data_dir", type=str, default="", help="Path to labelled dataset root")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--pretrained_weights", type=str, default="", help="Path to pretrained sequence model weights")
    args = parser.parse_args()

    dm = DatasetManager()
    
    # Check default Standard OCR dataset location or user-specified path
    data_path = args.data_dir
    if not data_path:
        default_std_path = os.path.join(os.path.dirname(BASE_DIR), "ocr", "data")
        if os.path.exists(default_std_path):
            data_path = default_std_path

    items = []
    if data_path and os.path.exists(data_path):
        print(f"Loading paired samples from {data_path}...")
        try:
            raw_items = dm.load_dataset(args.dataset_type, data_path)
            items, errors = dm.validate_and_sanitize(raw_items)
            print(f"Loaded {len(items)} valid samples ({len(errors)} invalid discarded).")
        except Exception as e:
            print(f"Warning: Failed loading dataset from {data_path}: {e}")

    # Honest failure condition: NEVER fabricate pseudo-labels or fake samples
    if not items:
        raise RuntimeError(
            "No labelled text-line dataset found. Provide real image crops and exact transcriptions "
            "via --data_dir or configure a valid dataset before launching training."
        )

    # Leak-free split grouped by parent document or publisher
    splits = dm.build_grouped_splits(
        items,
        group_key="source_document_id",
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10
    )
    print(f"Dataset split complete: {len(splits.train)} train, {len(splits.val)} val, {len(splits.test)} test.")

    cfg = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_checkpoint=args.resume if args.resume else None,
        pretrained_weights_path=args.pretrained_weights if args.pretrained_weights else None
    )

    pipeline = FineTunePipeline(cfg)
    pipeline.fit(splits.train, splits.val)

if __name__ == "__main__":
    main()
