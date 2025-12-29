"""
Cohen's Kappa inter-annotator agreement utilities.
Implements kappa calculation and disagreement analysis.
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter

def calculate_cohens_kappa(annotator1_terms: List[str], annotator2_terms: List[str]) -> Dict:
    """
    Calculate Cohen's Kappa between two annotators based on term lists.

    Args:
        annotator1_terms: terms from annotator 1
        annotator2_terms: terms from annotator 2

    Returns a dictionary with kappa, confusion matrix and stats.
    """
    # normalize terms (lowercase, strip)
    a1_terms = set(term.lower().strip() for term in annotator1_terms)
    a2_terms = set(term.lower().strip() for term in annotator2_terms)

    # create union of all observed terms
    all_terms = a1_terms.union(a2_terms)

    # Build a 2x2 confusion matrix based on set membership
    tp = len(a1_terms.intersection(a2_terms))  # True Positive: both annotated
    fp = len(a1_terms - a2_terms)  # False Positive: only annotator1
    fn = len(a2_terms - a1_terms)  # False Negative: only annotator2
    tn = len(all_terms) - tp - fp - fn  # True Negative: neither annotated

    # Observed agreement
    po = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # Expected agreement by chance
    total = tp + tn + fp + fn
    if total > 0:
        p_yes = ((tp + fp) / total) * ((tp + fn) / total)
        p_no = ((tn + fn) / total) * ((tn + fp) / total)
        pe = p_yes + p_no
    else:
        pe = 0

    # Cohen's Kappa
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0

    # Interpretation helper
    def interpret_kappa(k):
        if k <= 0:
            return "No agreement"
        elif k <= 0.20:
            return "Slight agreement"
        elif k <= 0.40:
            return "Fair agreement"
        elif k <= 0.60:
            return "Moderate agreement"
        elif k <= 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    return {
        "kappa": kappa,
        "interpretation": interpret_kappa(kappa),
        "observed_agreement": po,
        "expected_agreement": pe,
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        },
        "annotator_stats": {
            "annotator1_count": len(annotator1_terms),
            "annotator2_count": len(annotator2_terms),
            "agreement_count": len(a1_terms.intersection(a2_terms)),
            "unique_terms": len(all_terms)
        }
    }

def calculate_multiple_kappa(annotations: Dict[str, List[str]]) -> Dict:
    """
    Calculate pairwise Cohen's Kappa for multiple annotators.

    Args:
        annotations: mapping from annotator ID to their term lists

    Returns:
        Dictionary with all pairwise kappa results and the average.
    """
    annotators = list(annotations.keys())
    if len(annotators) < 2:
        return {"error": "Need at least 2 annotators for kappa calculation"}

    pairwise_kappas = {}
    kappa_values = []

    # Calculate all pairwise kappa
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a1 = annotators[i]
            a2 = annotators[j]
            kappa_result = calculate_cohens_kappa(annotations[a1], annotations[a2])
            pairwise_kappas[f"{a1}_vs_{a2}"] = kappa_result
            kappa_values.append(kappa_result["kappa"])

    # Calculate average kappa
    avg_kappa = np.mean(kappa_values) if kappa_values else 0

    return {
        "average_kappa": avg_kappa,
        "pairwise_kappas": pairwise_kappas,
        "annotator_count": len(annotators),
        "kappa_interpretation": interpret_kappa(avg_kappa)
    }

def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value using standard thresholds."""
    if kappa <= 0:
        return "No agreement"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def simulate_annotator_disagreement_analysis(gold_terms: List[str],
                                            predicted_terms: List[str]) -> Dict:
    """
    Simulate annotator disagreement by treating gold vs. predicted terms as two annotators.

    Args:
        gold_terms: list of gold-standard terms
        predicted_terms: list of predicted terms

    Returns:
        Disagreement analysis dictionary
    """
    kappa_result = calculate_cohens_kappa(gold_terms, predicted_terms)

    # Disagreement analysis by set operations
    gold_set = set(term.lower().strip() for term in gold_terms)
    pred_set = set(term.lower().strip() for term in predicted_terms)

    missing_terms = gold_set - pred_set
    extra_terms = pred_set - gold_set

    return {
        "kappa_analysis": kappa_result,
        "disagreement_analysis": {
            "missing_from_prediction": list(missing_terms),
            "extra_in_prediction": list(extra_terms),
            "missing_count": len(missing_terms),
            "extra_count": len(extra_terms),
            "agreement_count": len(gold_set.intersection(pred_set))
        },
        "agreement_rate": len(gold_set.intersection(pred_set)) / len(gold_set.union(pred_set)) if len(gold_set.union(pred_set)) > 0 else 0
    }

def load_sample_annotations(file_path: str = None) -> Dict:
    """
    Load sample annotation data (create simulated data if real multi-annotator data is absent).

    Args:
        file_path: optional path to annotation file

    Returns:
        Mapping from annotator ID to term lists
    """
    if file_path:
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Assume the data contains annotations from multiple annotators
            annotations = {}
            for i, doc in enumerate(data[:5]):  # take annotations for the first 5 documents
                if 'llm_annotated_terms' in doc:
                    annotations[f"annotator_{i+1}"] = doc['llm_annotated_terms']

            if len(annotations) >= 2:
                return annotations
        except Exception as e:
            print(f"Could not load annotations from {file_path}: {e}")

    # If real annotations are not available, return a small simulated set
    return {
        "annotator_1": [
            "cognitive bias", "attention", "memory", "emotion regulation",
            "social cognition", "decision making", "perception"
        ],
        "annotator_2": [
            "cognitive biases", "attentional processes", "working memory",
            "emotional regulation", "social perception", "decision-making",
            "visual perception", "reasoning"
        ],
        "annotator_3": [
            "bias", "attention", "long term memory", "emotion",
            "social thinking", "judgment", "perceptual processes"
        ]
    }