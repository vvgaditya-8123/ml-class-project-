import os
import json
import csv
from typing import List, Dict, Any, Optional

class BenchmarkReporter:
    """
    Exports comprehensive OCR evaluation reports to Markdown, CSV, and JSON.
    Guarantees that missing or uncalculated metrics are rendered as N/A rather than 0.00%.
    """

    @staticmethod
    def _format_pct(val: Optional[float]) -> str:
        if val is None:
            return "N/A"
        return f"{val * 100:.2f}%"

    @staticmethod
    def _format_float(val: Optional[float], decimals: int = 4) -> str:
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}"

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
            cer = BenchmarkReporter._format_pct(r.get("mean_cer"))
            wer = BenchmarkReporter._format_pct(r.get("mean_wer"))
            ned = BenchmarkReporter._format_pct(r.get("mean_ned"))
            em = BenchmarkReporter._format_pct(r.get("exact_match_accuracy"))
            f1 = BenchmarkReporter._format_float(r.get("mean_f1"), 4)
            lat = BenchmarkReporter._format_float(r.get("latency_ms"), 1)
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
