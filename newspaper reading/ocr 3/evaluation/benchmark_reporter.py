import os
import json
import csv
from typing import List, Dict, Any

class BenchmarkReporter:
    """
    Exports comprehensive OCR evaluation reports to Markdown, CSV, and JSON.
    """

    @staticmethod
    def to_markdown_table(results: List[Dict[str, Any]]) -> str:
        """Generates a GitHub-flavored Markdown comparison table."""
        headers = ["Model / Engine", "CER ↓", "WER ↓", "NED ↑", "Exact Match", "F1 Score", "Latency (ms)"]
        separator = ["| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]

        rows = []
        rows.append("| " + " | ".join(headers) + " |")
        rows.append(separator[0])

        for r in results:
            name = r.get("model_name", "Unknown")
            cer = f"{r.get('mean_cer', 0.0) * 100:.2f}%"
            wer = f"{r.get('mean_wer', 0.0) * 100:.2f}%"
            ned = f"{r.get('mean_ned', 0.0) * 100:.2f}%"
            em = f"{r.get('exact_match_accuracy', 0.0) * 100:.2f}%"
            f1 = f"{r.get('mean_f1', 0.0):.4f}"
            lat = f"{r.get('latency_ms', 0.0):.1f}"
            rows.append(f"| **{name}** | {cer} | {wer} | {ned} | {em} | {f1} | {lat} |")

        return "\n".join(rows)

    @staticmethod
    def export_csv(results: List[Dict[str, Any]], filepath: str):
        if not results:
            return
        keys = list(results[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Exported benchmark CSV -> {filepath}")

    @staticmethod
    def export_json(results: List[Dict[str, Any]], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Exported benchmark JSON -> {filepath}")

