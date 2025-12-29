"""
Enhanced Evaluation Metrics for Term Extraction
Includes Precision@K, MAP and other ranking metrics.
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import math

def calculate_precision_at_k(predicted_terms: List[str],
                           gold_terms: List[str],
                           k: int = 5) -> float:
    """
    Compute Precision@K: precision of the top-K predicted terms.

    Args:
        predicted_terms: ranked list of predicted terms
        gold_terms: list of gold-standard terms
        k: top-K cutoff

    Returns:
        Precision@K value
    """
    if k <= 0 or len(predicted_terms) == 0:
        return 0.0

    # take the top-K predicted terms
    top_k_predictions = predicted_terms[:k]

    # normalize terms
    gold_set = set(term.lower().strip() for term in gold_terms)

    # count how many of the top-K are correct
    correct_predictions = 0
    for term in top_k_predictions:
        if term.lower().strip() in gold_set:
            correct_predictions += 1

    return correct_predictions / k

def calculate_recall_at_k(predicted_terms: List[str],
                         gold_terms: List[str],
                         k: int = 10) -> float:
    """
    Compute Recall@K: fraction of gold terms covered by top-K predictions.

    Args:
        predicted_terms: ranked list of predicted terms
        gold_terms: list of gold-standard terms
        k: top-K cutoff

    Returns:
        Recall@K value
    """
    if len(gold_terms) == 0:
        return 0.0

    top_k_predictions = predicted_terms[:k] if k <= len(predicted_terms) else predicted_terms
    gold_set = set(term.lower().strip() for term in gold_terms)

    # count how many gold-standard terms are covered
    covered_gold = 0
    for term in top_k_predictions:
        if term.lower().strip() in gold_set:
            covered_gold += 1

    return covered_gold / len(gold_terms)

def calculate_average_precision(predicted_terms: List[str],
                               gold_terms: List[str]) -> float:
    """
    Compute Average Precision (AP) for ranked predictions.
    Measures quality of the ranking with respect to gold terms.

    Args:
        predicted_terms: ranked list of predicted terms
        gold_terms: list of gold-standard terms

    Returns:
        Average Precision value
    """
    if len(predicted_terms) == 0 or len(gold_terms) == 0:
        return 0.0

    gold_set = set(term.lower().strip() for term in gold_terms)
    precisions = []
    correct_so_far = 0

    for i, term in enumerate(predicted_terms):
        if term.lower().strip() in gold_set:
            correct_so_far += 1
            precision_at_i = correct_so_far / (i + 1)
            precisions.append(precision_at_i)

    return np.mean(precisions) if precisions else 0.0

def calculate_mean_average_precision(all_predictions: List[List[str]],
                                    all_gold_terms: List[List[str]]) -> float:
    """
    Compute Mean Average Precision (MAP) across documents.

    Args:
        all_predictions: list of prediction lists for all documents
        all_gold_terms: list of gold-term lists for all documents

    Returns:
        MAP value
    """
    if len(all_predictions) != len(all_gold_terms):
        raise ValueError("Predictions and gold terms must have the same length")

    aps = []
    for pred_terms, gold_terms in zip(all_predictions, all_gold_terms):
        ap = calculate_average_precision(pred_terms, gold_terms)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0

def calculate_normalized_dcg(predicted_terms: List[str],
                           gold_terms: List[str],
                           k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        predicted_terms: ranked list of predicted terms
        gold_terms: list of gold-standard terms
        k: top-K cutoff

    Returns:
        NDCG@K value
    """
    def calculate_dcg(terms: List[str], gold_set: Set[str], k: int) -> float:
        dcg = 0.0
        for i, term in enumerate(terms[:k]):
            if term.lower().strip() in gold_set:
                dcg += 1.0 / math.log2(i + 2)  # use i+2 because log2(1) = 0
        return dcg

    gold_set = set(term.lower().strip() for term in gold_terms)

    # actual DCG
    actual_dcg = calculate_dcg(predicted_terms, gold_set, k)

    # ideal DCG (all relevant terms ranked first)
    ideal_terms = [term for term in gold_terms if term.lower().strip() in gold_set]
    ideal_dcg = calculate_dcg(ideal_terms, gold_set, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def calculate_ranking_metrics(predicted_terms: List[str],
                            gold_terms: List[str],
                            k_values: List[int] = [1, 3, 5, 10]) -> Dict:
    """
    Compute a suite of ranking metrics for a single document.

    Args:
        predicted_terms: ranked list of predicted terms
        gold_terms: list of gold-standard terms
        k_values: list of K values to evaluate

    Returns:
        Dictionary containing multiple ranking metrics
    """
    results = {
        "basic_metrics": calculate_basic_metrics(predicted_terms, gold_terms)
    }

    # Precision@K for different K values
    results["precision_at_k"] = {}
    for k in k_values:
        if k <= len(predicted_terms):
            results["precision_at_k"][f"P@{k}"] = calculate_precision_at_k(
                predicted_terms, gold_terms, k
            )

    # Recall@K for different K values
    results["recall_at_k"] = {}
    for k in k_values:
        results["recall_at_k"][f"R@{k}"] = calculate_recall_at_k(
            predicted_terms, gold_terms, k
        )

    # Average Precision and NDCG
    results["ap"] = calculate_average_precision(predicted_terms, gold_terms)
    results["ndcg"] = {}
    for k in k_values:
        if k <= len(predicted_terms):
            results["ndcg"][f"NDCG@{k}"] = calculate_normalized_dcg(
                predicted_terms, gold_terms, k
            )

    # Reciprocal Rank (RR): reciprocal of the rank of the first correct answer
    rr = 0.0
    for i, term in enumerate(predicted_terms):
        if term.lower().strip() in set(g.lower().strip() for g in gold_terms):
            rr = 1.0 / (i + 1)
            break
    results["reciprocal_rank"] = rr

    return results

def calculate_basic_metrics(predicted_terms: List[str],
                          gold_terms: List[str]) -> Dict:
    """
    Compute basic precision, recall and F1 metrics.

    Args:
        predicted_terms: list of predicted terms
        gold_terms: list of gold-standard terms

    Returns:
        Dictionary of basic evaluation metrics
    """
    pred_set = set(p.lower().strip() for p in predicted_terms)
    gold_set = set(g.lower().strip() for g in gold_terms)

    true_positives = len(pred_set.intersection(gold_set))
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(gold_set) if len(gold_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_count": len(predicted_terms),
        "gold_count": len(gold_terms)
    }

def evaluate_system_performance(system_predictions: List[List[str]],
                              gold_standard: List[List[str]],
                              system_name: str = "System") -> Dict:
    """
    Evaluate performance of a system across all documents.

    Args:
        system_predictions: list of prediction lists for each document
        gold_standard: list of gold-term lists for each document
        system_name: optional system name label

    Returns:
        Dictionary containing aggregated system evaluation results
    """
    if len(system_predictions) != len(gold_standard):
        raise ValueError("Number of predictions must match number of gold standard annotations")

    # collect metrics for all documents
    all_precision_at_k = defaultdict(list)
    all_recall_at_k = defaultdict(list)
    all_ap = []
    all_basic_metrics = []

    k_values = [1, 3, 5, 10]

    for pred_terms, gold_terms in zip(system_predictions, gold_standard):
        if not gold_terms:  # skip documents without gold-standard annotations
            continue

        # compute basic metrics
        basic_metrics = calculate_basic_metrics(pred_terms, gold_terms)
        all_basic_metrics.append(basic_metrics)

        # compute ranking metrics
        ranking_metrics = calculate_ranking_metrics(pred_terms, gold_terms, k_values)

        # collect Precision@K
        for k in k_values:
            if f"P@{k}" in ranking_metrics["precision_at_k"]:
                all_precision_at_k[f"P@{k}"].append(ranking_metrics["precision_at_k"][f"P@{k}"])

        # collect Recall@K
        for k in k_values:
            if f"R@{k}" in ranking_metrics["recall_at_k"]:
                all_recall_at_k[f"R@{k}"].append(ranking_metrics["recall_at_k"][f"R@{k}"])

        # collect AP
        all_ap.append(ranking_metrics["ap"])

    # compute average metrics
    avg_precision_at_k = {}
    for k, values in all_precision_at_k.items():
        avg_precision_at_k[k] = np.mean(values) if values else 0.0

    avg_recall_at_k = {}
    for k, values in all_recall_at_k.items():
        avg_recall_at_k[k] = np.mean(values) if values else 0.0

    # compute average basic metrics
    avg_basic_metrics = {}
    if all_basic_metrics:
        for metric_name in all_basic_metrics[0].keys():
            if metric_name in ["precision", "recall", "f1"]:
                avg_basic_metrics[f"avg_{metric_name}"] = np.mean([
                    m[metric_name] for m in all_basic_metrics
                ])

    return {
        "system_name": system_name,
        "documents_evaluated": len(all_basic_metrics),
        "average_precision_at_k": avg_precision_at_k,
        "average_recall_at_k": avg_recall_at_k,
        "map_score": np.mean(all_ap) if all_ap else 0.0,
        "basic_metrics": avg_basic_metrics,
        "detailed_results": {
            "precision_at_k_values": all_precision_at_k,
            "recall_at_k_values": all_recall_at_k,
            "ap_values": all_ap
        }
    }