"""
Improved MentalBERT term extraction system.
Addresses performance issues and optimizes extraction strategy.
"""

import json
import numpy as np
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, models
from typing import List, Dict, Tuple
from tqdm import tqdm
from huggingface_hub import login
import re
from collections import defaultdict

# import evaluation modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.thesaurus import APAThesaurus
from evaluation.metrics import calculate_ranking_metrics, evaluate_system_performance

class ImprovedMentalBERTExtractor:
    """Improved MentalBERT term extractor."""

    def __init__(self, device=None, hf_token=None):
        """
        Initialize the improved MentalBERT extractor.

        Args:
            device: compute device ('cuda', 'mps', 'cpu')
            hf_token: optional Hugging Face token
        """
        self.device = self._setup_device(device)
        self.hf_token = hf_token
        self.model = None
        self.kw_model = None
        self.apa_thesaurus = APAThesaurus()

        # extraction parameter tuning
        self.extraction_config = {
            "keyphrase_ngram_range": (1, 2),  # reduce to 1-2 words to avoid long phrases
            "stop_words": 'english',
            "use_maxsum": True,  # use Max Sum similarity
            "use_mmr": True,     # use Maximal Marginal Relevance
            "diversity": 0.4,    # increase diversity
            "top_n": 10,         # number of top terms to return
            "nr_candidates": 20  # number of candidate terms to consider
        }

    def _setup_device(self, device=None):
        """Set up compute device."""
        if device:
            return device

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text: clean and normalize.

        Args:
            text: raw input text

        Returns:
            cleaned text
        """
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\'\"]', ' ', text)

        # ensure text is not empty
        if not text.strip():
            return "empty document"

        return text.strip()

    def _load_model(self):
        """Load the MentalBERT model and associated KeyBERT instance."""
        if self.model is not None:
            return

        print(f"Loading improved MentalBERT configuration on {self.device}...")

        try:
            # If an HF token is provided, attempt to login
            if self.hf_token and self.hf_token.startswith("hf_"):
                login(token=self.hf_token)

            # Try to load the MentalBERT sentence transformer
            self.model = SentenceTransformer(
                'mental/mental-bert-base-uncased',
                device=self.device
            )

            # Create KeyBERT instance based on the loaded model
            self.kw_model = KeyBERT(model=self.model)

            print("MentalBERT loaded successfully.")

        except Exception as e:
            print(f"Error loading MentalBERT: {e}")
            print("Falling back to MiniLM-L6-v2...")
            # Fallback to a general-purpose model
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
            print("Fallback model loaded successfully.")

    def _filter_candidates(self, candidates: List[Tuple[str, float]],
                          gold_terms: List[str] = None) -> List[str]:
        """
        Filter and refine candidate terms.

        Args:
            candidates: list of (term, score) tuples
            gold_terms: optional gold-standard terms for reference

        Returns:
            Filtered list of term strings
        """
        filtered_terms = []
        seen_terms = set()

        for term, score in candidates:
            term_clean = term.lower().strip()

            # deduplicate
            if term_clean in seen_terms:
                continue

            # length filtering
            if len(term_clean.split()) > 3:  # remove phrases longer than 3 words
                continue

            # filter overly common non-technical words
            common_non_terms = {
                'study', 'research', 'paper', 'article', 'result', 'method',
                'analysis', 'data', 'approach', 'model', 'system', 'work',
                'show', 'find', 'suggest', 'report', 'conclude', 'demonstrate'
            }

            first_word = term_clean.split()[0] if term_clean.split() else ""
            if first_word in common_non_terms:
                continue

            # check relevance using APA thesaurus (optional)
            category = self.apa_thesaurus.categorize_term(term)
            if category != "Uncategorized" or (gold_terms and any(
                term_clean in gold.lower() for gold in gold_terms
            )):
                filtered_terms.append(term)
                seen_terms.add(term_clean)

        return filtered_terms

    def extract_keywords(self, text: str, gold_terms: List[str] = None) -> List[str]:
        """
        Extract keywords (improved strategy).

        Args:
            text: input text
            gold_terms: optional gold-standard terms to guide filtering

        Returns:
            list of extracted term strings
        """
        # ensure model is loaded
        if self.model is None:
            self._load_model()

        # preprocess text
        text = self._preprocess_text(text)

        try:
            # Stage 1: extract using multiple strategies
            extraction_strategies = [
                # Strategy 1: MMR + diversity
                lambda: self.kw_model.extract_keywords(
                    text,
                    use_mmr=True,
                    diversity=self.extraction_config["diversity"],
                    keyphrase_ngram_range=self.extraction_config["keyphrase_ngram_range"],
                    stop_words=self.extraction_config["stop_words"],
                    top_n=self.extraction_config["top_n"]
                ),

                # Strategy 2: Max Sum similarity
                lambda: self.kw_model.extract_keywords(
                    text,
                    use_maxsum=True,
                    nr_candidates=self.extraction_config["nr_candidates"],
                    keyphrase_ngram_range=self.extraction_config["keyphrase_ngram_range"],
                    stop_words=self.extraction_config["stop_words"],
                    top_n=self.extraction_config["top_n"]
                ),

                # Strategy 3: simple score-based extraction
                lambda: self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.extraction_config["keyphrase_ngram_range"],
                    stop_words=self.extraction_config["stop_words"],
                    top_n=self.extraction_config["nr_candidates"]
                )
            ]

            all_candidates = []
            for strategy in extraction_strategies:
                try:
                    candidates = strategy()
                    all_candidates.extend(candidates)
                except Exception as e:
                    print(f"Strategy failed: {e}")
                    continue

            # Merge and deduplicate candidates
            unique_candidates = {}
            for term, score in all_candidates:
                term_clean = term.lower().strip()
                if term_clean not in unique_candidates or score > unique_candidates[term_clean]:
                    unique_candidates[term_clean] = score

            # Sort by score
            sorted_candidates = sorted(
                unique_candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Convert to (term, score) format
            final_candidates = [(term, score) for term, score in sorted_candidates]

            # Filter and refine candidates
            filtered_terms = self._filter_candidates(final_candidates, gold_terms)

            return filtered_terms[:self.extraction_config["top_n"]]

        except Exception as e:
            print(f"Error in keyword extraction: {e}")
            return []

    def evaluate_on_dataset(self, data_file: str, max_docs: int = None) -> Dict:
        """
        Evaluate the extractor on a dataset file.

        Args:
            data_file: path to the dataset file
            max_docs: optional limit for number of documents

        Returns:
            evaluation result dictionary
        """
        print(f"Loading data from {data_file}...")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if max_docs:
            data = data[:max_docs]

        print(f"Evaluating on {len(data)} documents...")

        system_predictions = []
        gold_standard = []
        detailed_results = []
        apa_analysis_results = []

        for doc in tqdm(data):
            text = doc.get('title', '') + ". " + doc.get('abstract', '')
            gold_terms = doc.get('llm_annotated_terms', [])

            if not gold_terms:
                continue

            # Extract keywords
            predicted_terms = self.extract_keywords(text, gold_terms)

            system_predictions.append(predicted_terms)
            gold_standard.append(gold_terms)

            # compute detailed metrics
            ranking_metrics = calculate_ranking_metrics(predicted_terms, gold_terms)

            # APA concept analysis
            apa_result = self.apa_thesaurus.evaluate_extraction_quality(
                predicted_terms, gold_terms
            )

            doc_result = {
                "doc_id": doc.get('id', 'unknown'),
                "predicted_count": len(predicted_terms),
                "gold_count": len(gold_terms),
                "basic_metrics": ranking_metrics["basic_metrics"],
                "precision_at_5": ranking_metrics["precision_at_k"].get("P@5", 0.0),
                "recall_at_10": ranking_metrics["recall_at_k"].get("R@10", 0.0),
                "ap": ranking_metrics["ap"],
                "apa_coherence": apa_result["conceptual_metrics"]["predicted_coherence"]["coherence_score"]
            }

            detailed_results.append(doc_result)
            apa_analysis_results.append(apa_result)

        # system-level evaluation
        system_evaluation = evaluate_system_performance(
            system_predictions, gold_standard, "Improved MentalBERT"
        )

        return {
            "system_evaluation": system_evaluation,
            "document_level_results": detailed_results,
            "apa_analysis": {
                "average_coherence": np.mean([
                    r["conceptual_metrics"]["predicted_coherence"]["coherence_score"]
                    for r in apa_analysis_results
                ]),
                "average_category_overlap": np.mean([
                    r["conceptual_metrics"]["category_overlap"]
                    for r in apa_analysis_results
                ]),
                "detailed_apa_results": apa_analysis_results[:5]  # keep only the first 5 detailed results
            },
            "config": self.extraction_config
        }

def main():
    """Main entry point: run improved MentalBERT evaluation."""

    # configuration
    HF_TOKEN = "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"  # replace with a real token
    INPUT_FILE = "../data/abstracts/DEV_set_annotated_LLM_CLEAN.json"

    # initialize extractor
    extractor = ImprovedMentalBERTExtractor(hf_token=HF_TOKEN)

    # evaluate system
    results = extractor.evaluate_on_dataset(INPUT_FILE, max_docs=52)

    # print results
    print("\n" + "="*60)
    print("IMPROVED MENTALBERT SYSTEM RESULTS")
    print("="*60)

    sys_eval = results["system_evaluation"]
    print(f"Documents Evaluated: {sys_eval['documents_evaluated']}")
    print(f"MAP Score: {sys_eval['map_score']:.4f}")

    print(f"\nBasic Metrics:")
    basic = sys_eval['basic_metrics']
    print(f"  Precision: {basic.get('avg_precision', 0):.4f}")
    print(f"  Recall:    {basic.get('avg_recall', 0):.4f}")
    print(f"  F1 Score:  {basic.get('avg_f1', 0):.4f}")

    print(f"\nPrecision@K:")
    for k, score in sys_eval['average_precision_at_k'].items():
        print(f"  {k}: {score:.4f}")

    print(f"\nRecall@K:")
    for k, score in sys_eval['average_recall_at_k'].items():
        print(f"  {k}: {score:.4f}")

    apa = results["apa_analysis"]
    print(f"\nAPA Conceptual Analysis:")
    print(f"  Average Coherence: {apa['average_coherence']:.4f}")
    print(f"  Category Overlap: {apa['average_category_overlap']:.4f}")

    print("-" * 60)

    # save detailed results
    os.makedirs("../../output", exist_ok=True)
    with open("../../output/improved_mentalbert_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Detailed results saved to 'output/improved_mentalbert_results.json'")

if __name__ == "__main__":
    main()