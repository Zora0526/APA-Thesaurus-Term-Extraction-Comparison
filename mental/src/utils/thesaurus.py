"""
APA Thesaurus Integration Module for Conceptual Analysis
Implements conceptual analysis and APA thesaurus alignment.
"""

import json
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class APAThesaurus:
    """
    APA psychology thesaurus helper for conceptual analysis and alignment.
    """

    def __init__(self):
        # Main APA thesaurus categories (based on APA Thesaurus 12th Edition)
        self.apa_categories = {
            "Emotion Processes": [
                "emotion", "emotional regulation", "affect", "mood", "anxiety", "depression",
                "stress", "coping", "emotional intelligence", "fear", "anger", "happiness",
                "emotional expression", "emotional experience", "affective disorders"
            ],
            "Cognitive Mechanisms": [
                "cognition", "attention", "memory", "perception", "thinking", "reasoning",
                "problem solving", "decision making", "cognitive bias", "learning", "judgment",
                "information processing", "cognitive load", "working memory", "long term memory"
            ],
            "Social Psychology": [
                "social cognition", "attitude", "stereotype", "prejudice", "group dynamics",
                "social influence", "interpersonal relations", "social behavior", "identity",
                "self concept", "social perception", "attribution", "conformity", "obedience"
            ],
            "Developmental Psychology": [
                "development", "child development", "adolescent development", "aging",
                "lifespan development", "developmental stages", "cognitive development",
                "social development", "emotional development", "attachment", "developmental disorders"
            ],
            "Clinical Psychology": [
                "mental disorders", "psychopathology", "therapy", "treatment", "diagnosis",
                "clinical assessment", "psychological intervention", "counseling", "psychotherapy",
                "behavioral disorders", "psychological well-being", "mental health"
            ],
            "Personality Psychology": [
                "personality", "personality traits", "personality assessment", "character",
                "temperament", "big five", "personality development", "self esteem", "identity",
                "personality disorders", "individual differences"
            ],
            "Neuropsychology": [
                "brain", "neural", "cognitive neuroscience", "neuropsychological assessment",
                "brain function", "cognitive control", "executive function", "neurotransmitters",
                "brain imaging", "neural networks", "neuroplasticity"
            ],
            "Methodology": [
                "research methods", "statistical analysis", "experimental design", "measurement",
                "assessment", "evaluation", "data analysis", "research design", "methodology",
                "psychometrics", "validity", "reliability", "statistical significance"
            ]
        }

        # Create a mapping from term to APA category
        self.term_to_category = {}
        for category, terms in self.apa_categories.items():
            for term in terms:
                self.term_to_category[term.lower()] = category

        # Expand term mapping with variants and synonyms
        self._expand_term_mapping()

    def _expand_term_mapping(self):
        """Extend term mapping with plural forms and common variants."""
        expansions = {
            "emotion": ["emotions", "emotional", "emotionally"],
            "cognition": ["cognitive", "cognitions"],
            "anxiety": ["anxious", "anxiety disorders"],
            "depression": ["depressive", "depressed", "major depression"],
            "stress": ["stressful", "stressors"],
            "memory": ["memories", "memorial", "remembering"],
            "learning": ["learn", "learned", "learning processes"],
            "attention": ["attentive", "attentional", "attention span"],
            "perception": ["perceptual", "perceive", "perceiving"],
            "therapy": ["psychotherapy", "counseling", "treatment"],
            "self concept": ["self-concept", "selfconcept", "self perception"],
            "self esteem": ["self-esteem", "selfesteem", "self worth"],
        }

        for base_term, variants in expansions.items():
            if base_term.lower() in self.term_to_category:
                category = self.term_to_category[base_term.lower()]
                for variant in variants:
                    self.term_to_category[variant.lower()] = category

    def categorize_term(self, term: str) -> str:
        """
        Categorize a term into an APA thesaurus category.

        Returns the category name or 'Uncategorized' if no match is found.
        """
        term_lower = term.lower().strip()

        # Direct match
        if term_lower in self.term_to_category:
            return self.term_to_category[term_lower]

        # Partial match: check if known keywords appear in the term
        for category, terms in self.apa_categories.items():
            for t in terms:
                if t.lower() in term_lower or term_lower in t.lower():
                    return category

        # Fuzzy match based on heuristic keywords
        keywords_mapping = {
            "Emotion Processes": ["emotion", "affect", "feeling", "mood"],
            "Cognitive Mechanisms": ["cognit", "think", "memory", "attention", "percept"],
            "Social Psychology": ["social", "interpersonal", "group", "attitude"],
            "Developmental Psychology": ["development", "child", "adolescent", "aging"],
            "Clinical Psychology": ["clinical", "disorder", "therapy", "treatment"],
            "Personality Psychology": ["personality", "trait", "character", "self"],
            "Neuropsychology": ["neural", "brain", "neuro", "cortical"],
            "Methodology": ["research", "method", "statistic", "measure", "assess"]
        }

        for category, keywords in keywords_mapping.items():
            for keyword in keywords:
                if keyword in term_lower:
                    return category

        return "Uncategorized"

    def calculate_conceptual_coherence(self, terms: List[str]) -> Dict:
        """
        Compute conceptual coherence for a list of terms.

        Returns a dictionary with coherence score and distribution.
        """
        if not terms:
            return {"coherence_score": 0.0, "category_distribution": {}, "dominant_category": None}

        categories = [self.categorize_term(term) for term in terms]
        category_counts = defaultdict(int)

        for category in categories:
            category_counts[category] += 1

        # Compute the proportion of the dominant category
        total_terms = len(terms)
        dominant_count = max(category_counts.values()) if category_counts else 0
        coherence_score = dominant_count / total_terms if total_terms > 0 else 0.0

        # Convert distribution to percentage
        category_distribution = {
            category: count / total_terms * 100
            for category, count in category_counts.items()
        }

        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None

        return {
            "coherence_score": coherence_score,
            "category_distribution": category_distribution,
            "dominant_category": dominant_category,
            "category_counts": dict(category_counts)
        }

    def evaluate_extraction_quality(self, predicted_terms: List[str], gold_terms: List[str]) -> Dict:
        """
        Evaluate extraction quality using APA thesaurus alignment.

        Returns exact-match metrics and conceptual alignment metrics.
        """
        # Basic exact-match metrics
        predicted_set = set(p.lower().strip() for p in predicted_terms)
        gold_set = set(g.lower().strip() for g in gold_terms)

        true_positives = len(predicted_set.intersection(gold_set))
        precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0
        recall = true_positives / len(gold_set) if len(gold_set) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Conceptual relevance analysis
        predicted_categories = [self.categorize_term(term) for term in predicted_terms]
        gold_categories = [self.categorize_term(term) for term in gold_terms]

        # Compute category overlap
        pred_category_set = set(predicted_categories)
        gold_category_set = set(gold_categories)
        category_overlap = len(pred_category_set.intersection(gold_category_set)) / len(gold_category_set.union(pred_category_set)) if len(gold_category_set.union(pred_category_set)) > 0 else 0.0

        # Conceptual coherence
        pred_coherence = self.calculate_conceptual_coherence(predicted_terms)
        gold_coherence = self.calculate_conceptual_coherence(gold_terms)

        return {
            "exact_match_metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "predicted_count": len(predicted_terms),
                "gold_count": len(gold_terms)
            },
            "conceptual_metrics": {
                "category_overlap": category_overlap,
                "predicted_coherence": pred_coherence,
                "gold_coherence": gold_coherence,
                "coherence_preservation": min(pred_coherence["coherence_score"] / gold_coherence["coherence_score"], 1.0) if gold_coherence["coherence_score"] > 0 else 0.0
            },
            "category_analysis": {
                "predicted_categories": dict(Counter(predicted_categories)),
                "gold_categories": dict(Counter(gold_categories))
            }
        }

from collections import Counter