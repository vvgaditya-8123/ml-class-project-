import numpy as np
from typing import List, Tuple, Dict, Any

def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Computes minimum edit distance (Levenshtein) between two sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return int(dp[n][m])

def character_error_rate(ref: str, hyp: str) -> float:
    """CER = Levenshtein(chars(ref), chars(hyp)) / len(chars(ref))"""
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return min(1.0, dist / float(len(ref_chars)))

def word_error_rate(ref: str, hyp: str) -> float:
    """WER = Levenshtein(words(ref), words(hyp)) / len(words(ref))"""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return min(1.0, dist / float(len(ref_words)))

def normalized_edit_distance(ref: str, hyp: str) -> float:
    """NED = 1 - (Levenshtein(ref, hyp) / max(len(ref), len(hyp)))"""
    max_len = max(len(ref), len(hyp))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(list(ref), list(hyp))
    return max(0.0, 1.0 - (dist / float(max_len)))

def token_precision_recall_f1(ref: str, hyp: str) -> Dict[str, float]:
    """Token-level precision, recall, and harmonic F1 score."""
    ref_tokens = ref.strip().split()
    hyp_tokens = hyp.strip().split()

    if not ref_tokens and not hyp_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    hyp_counts = {}
    for t in hyp_tokens:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1

    for t, count in hyp_counts.items():
        overlap += min(count, ref_counts.get(t, 0))

    prec = overlap / float(len(hyp_tokens))
    rec = overlap / float(len(ref_tokens))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

def layout_reading_order_score(true_order_ids: List[int], pred_order_ids: List[int]) -> float:
    """
    Evaluates layout-aware reading order correctness using Kendall's Tau correlation:
    Measures concordant vs discordant reading block sequence pairs.
    Returns score in [0.0, 1.0].
    """
    n = len(true_order_ids)
    if n <= 1:
        return 1.0

    # Map item to its rank in predicted order
    pred_ranks = {item_id: r for r, item_id in enumerate(pred_order_ids)}
    
    concordant = 0
    discordant = 0
    total_pairs = (n * (n - 1)) // 2

    for i in range(n):
        for j in range(i + 1, n):
            id_a = true_order_ids[i]
            id_b = true_order_ids[j]

            if id_a in pred_ranks and id_b in pred_ranks:
                if pred_ranks[id_a] < pred_ranks[id_b]:
                    concordant += 1
                else:
                    discordant += 1
            else:
                discordant += 1

    if total_pairs == 0:
        return 1.0
    # Normalized from [-1, 1] to [0, 1]
    tau = (concordant - discordant) / float(total_pairs)
    return max(0.0, (tau + 1.0) / 2.0)

def compute_ocr_metrics_suite(references: List[str], hypotheses: List[str]) -> Dict[str, Any]:
    """Computes comprehensive OCR evaluation metrics across a dataset."""
    assert len(references) == len(hypotheses), "References and Hypotheses counts must match"
    
    cers = []
    wers = []
    neds = []
    exact_matches = 0
    precisions = []
    recalls = []
    f1s = []

    for ref, hyp in zip(references, hypotheses):
        cers.append(character_error_rate(ref, hyp))
        wers.append(word_error_rate(ref, hyp))
        neds.append(normalized_edit_distance(ref, hyp))
        if ref.strip() == hyp.strip():
            exact_matches += 1

        prf = token_precision_recall_f1(ref, hyp)
        precisions.append(prf["precision"])
        recalls.append(prf["recall"])
        f1s.append(prf["f1"])

    total = max(1, len(references))
    return {
        "num_samples": len(references),
        "mean_cer": round(float(np.mean(cers)), 4),
        "mean_wer": round(float(np.mean(wers)), 4),
        "mean_ned": round(float(np.mean(neds)), 4),
        "exact_match_accuracy": round(float(exact_matches / total), 4),
        "mean_precision": round(float(np.mean(precisions)), 4),
        "mean_recall": round(float(np.mean(recalls)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4)
    }
