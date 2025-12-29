"""
Systematic Error Analysis for Term Extraction
Implements cross-subfield error analysis required by the project proposal.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.thesaurus import APAThesaurus

class TermExtractionErrorAnalyzer:
    """Term extraction error analyzer."""

    def __init__(self):
        self.apa_thesaurus = APAThesaurus()
        self.error_categories = {
            "Missing Core Concepts": [],      # missing essential/core concepts
            "Extracting Generic Terms": [],   # extracted overly generic terms
            "Partial Matches": [],            # partial matches
            "Over-specific Terms": [],        # overly specific terms
            "Semantic Drift": [],             # semantic drift
            "Category Misclassification": []  # misclassification across categories
        }

    def categorize_error_type(self, gold_term: str, predicted_terms: List[str]) -> List[str]:
        """
        Analyze the error types for a gold-standard term given predicted terms.

        Args:
            gold_term: gold-standard term string
            predicted_terms: list of predicted term strings

        Returns:
            A list of detected error type labels for the gold term.
        """
        errors = []
        gold_lower = gold_term.lower().strip()
        pred_lower = [p.lower().strip() for p in predicted_terms]

        # check if the gold term was predicted exactly
        is_predicted = any(gold_lower == p for p in pred_lower)

        if is_predicted:
            return errors  # no error

        # check for partial matches
        partial_matches = []
        for pred in pred_lower:
            if gold_lower in pred or pred in gold_lower:
                partial_matches.append(pred)

        if partial_matches:
            errors.append("Partial Matches")

        # check whether the gold term belongs to a core category that was missed
        gold_category = self.apa_thesaurus.categorize_term(gold_term)
        predicted_categories = [self.apa_thesaurus.categorize_term(p) for p in predicted_terms]
        # if the gold term has a category but no predictions fall into that category
        if gold_category != "Uncategorized":
            if gold_category not in predicted_categories:
                errors.append("Missing Core Concepts")

        # check semantic relevance between predictions and gold category
        has_semantic_relevance = False
        for pred in predicted_terms:
            pred_category = self.apa_thesaurus.categorize_term(pred)
            if pred_category == gold_category and pred_category != "Uncategorized":
                has_semantic_relevance = True
                break

        if not has_semantic_relevance and gold_category != "Uncategorized":
            errors.append("Semantic Drift")

        return errors if errors else ["Other Missing"]

    def analyze_document_errors(self, gold_terms: List[str],
                              predicted_terms: List[str],
                              doc_metadata: Dict = None) -> Dict:
        """
        Analyze errors for a single document.

        Args:
            gold_terms: list of gold-standard terms
            predicted_terms: list of predicted terms
            doc_metadata: optional document metadata (subfield, source, etc.)

        Returns:
            Document-level error analysis result dictionary.
        """
        doc_errors = {
            "missing_terms": [],
            "extra_terms": [],
            "error_types": defaultdict(list),
            "subfield_analysis": {}
        }

        gold_set = set(term.lower().strip() for term in gold_terms)
        pred_set = set(term.lower().strip() for term in predicted_terms)

        # analyze missing (gold) terms
        for gold_term in gold_terms:
            if gold_term.lower().strip() not in pred_set:
                errors = self.categorize_error_type(gold_term, predicted_terms)
                doc_errors["missing_terms"].append({
                    "term": gold_term,
                    "errors": errors,
                    "category": self.apa_thesaurus.categorize_term(gold_term)
                })

                for error_type in errors:
                    doc_errors["error_types"][error_type].append(gold_term)

        # analyze extra (predicted-only) terms
        for pred_term in predicted_terms:
            if pred_term.lower().strip() not in gold_set:
                pred_category = self.apa_thesaurus.categorize_term(pred_term)

                if pred_category == "Uncategorized":
                    doc_errors["extra_terms"].append({
                        "term": pred_term,
                        "category": pred_category,
                        "likely_error": "Extracting Generic Terms"
                    })
                else:
                    doc_errors["extra_terms"].append({
                        "term": pred_term,
                        "category": pred_category,
                        "likely_error": "Over-specific Terms"
                    })

        # subfield-level analysis
        if doc_metadata:
            subfield = doc_metadata.get('subfield', 'Unknown')
            doc_errors["subfield_analysis"] = {
                "subfield": subfield,
                "gold_terms_by_category": self._categorize_terms(gold_terms),
                "predicted_terms_by_category": self._categorize_terms(predicted_terms)
            }

        return doc_errors

    def _categorize_terms(self, terms: List[str]) -> Dict:
        """Categorize terms by APA thesaurus categories."""
        categories = defaultdict(list)
        for term in terms:
            category = self.apa_thesaurus.categorize_term(term)
            categories[category].append(term)
        return dict(categories)

    def analyze_subfield_performance(self, documents: List[Dict]) -> Dict:
        """
        Analyze performance across psychological subfields.

        Args:
            documents: list of documents including gold terms, predictions, and metadata

        Returns:
            Per-subfield performance analysis dictionary.
        """
        subfield_performance = defaultdict(lambda: {
            "documents": 0,
            "total_gold_terms": 0,
            "total_predicted_terms": 0,
            "correct_predictions": 0,
            "missing_core_concepts": 0,
            "generic_terms_extracted": 0,
            "partial_matches": 0,
            "error_types": defaultdict(int),
            "category_coherence": []
        })

        for doc in documents:
            # determine subfield (based on metadata or document text)
            subfield = self._determine_subfield(doc)
            performance = subfield_performance[subfield]

            performance["documents"] += 1

            gold_terms = doc.get('gold_terms', [])
            predicted_terms = doc.get('predicted_terms', [])

            performance["total_gold_terms"] += len(gold_terms)
            performance["total_predicted_terms"] += len(predicted_terms)

            # compute number of correct predictions
            gold_set = set(g.lower().strip() for g in gold_terms)
            pred_set = set(p.lower().strip() for p in predicted_terms)
            correct = len(gold_set.intersection(pred_set))
            performance["correct_predictions"] += correct

            # perform error analysis
            error_analysis = self.analyze_document_errors(gold_terms, predicted_terms, doc)

            for error_type, terms in error_analysis["error_types"].items():
                performance["error_types"][error_type] += len(terms)

            # category coherence analysis
            gold_coherence = self.apa_thesaurus.calculate_conceptual_coherence(gold_terms)
            pred_coherence = self.apa_thesaurus.calculate_conceptual_coherence(predicted_terms)

            performance["category_coherence"].append({
                "gold_coherence": gold_coherence["coherence_score"],
                "pred_coherence": pred_coherence["coherence_score"],
                "dominant_gold_category": gold_coherence["dominant_category"],
                "dominant_pred_category": pred_coherence["dominant_category"]
            })

        # compute aggregate metrics
        results = {}
        for subfield, performance in subfield_performance.items():
            if performance["documents"] > 0:
                precision = performance["correct_predictions"] / performance["total_predicted_terms"] if performance["total_predicted_terms"] > 0 else 0
                recall = performance["correct_predictions"] / performance["total_gold_terms"] if performance["total_gold_terms"] > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                avg_gold_coherence = np.mean([c["gold_coherence"] for c in performance["category_coherence"]])
                avg_pred_coherence = np.mean([c["pred_coherence"] for c in performance["category_coherence"]])

                results[subfield] = {
                    "document_count": performance["documents"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "avg_gold_coherence": avg_gold_coherence,
                    "avg_pred_coherence": avg_pred_coherence,
                    "coherence_preservation": avg_pred_coherence / avg_gold_coherence if avg_gold_coherence > 0 else 0,
                    "error_distribution": dict(performance["error_types"]),
                    "total_errors": sum(performance["error_types"].values())
                }

        return results

    def _determine_subfield(self, doc: Dict) -> str:
        """
        Determine the psychology subfield for a document based on metadata or text.

        Args:
            doc: document data dictionary

        Returns:
            Subfield name string.
        """
        # first check metadata
        if 'subfield' in doc:
            return doc['subfield']

        # determine subfield using title and abstract keywords
        text = f"{doc.get('title', '')} {doc.get('abstract', '')}".lower()

        subfield_keywords = {
            "Cognitive Psychology": ["cognitive", "memory", "attention", "perception", "thinking", "reasoning"],
            "Social Psychology": ["social", "interpersonal", "group", "attitude", "stereotype", "prejudice"],
            "Clinical Psychology": ["clinical", "disorder", "therapy", "treatment", "mental health", "diagnosis"],
            "Developmental Psychology": ["development", "child", "adolescent", "aging", "lifespan", "developmental"],
            "Personality Psychology": ["personality", "trait", "self", "character", "individual differences"],
            "Neuropsychology": ["neural", "brain", "neuro", "cognitive neuroscience", "neuropsychological"],
            "Emotion Psychology": ["emotion", "affect", "mood", "emotional", "feeling"]
        }

        subfield_scores = {}
        for subfield, keywords in subfield_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            subfield_scores[subfield] = score

        if subfield_scores:
            return max(subfield_scores, key=subfield_scores.get)

        return "General Psychology"

    def generate_error_report(self, documents: List[Dict], output_file: str = None) -> Dict:
        """
        Generate a comprehensive error analysis report for a list of documents.

        Args:
            documents: list of document dicts
            output_file: optional output file path to save the report

        Returns:
            A dictionary containing the error analysis report.
        """
        print("Generating comprehensive error analysis...")

        # global error statistics
        global_errors = {
            "total_documents": len(documents),
            "total_gold_terms": sum(len(doc.get('gold_terms', [])) for doc in documents),
            "total_predicted_terms": sum(len(doc.get('predicted_terms', [])) for doc in documents),
            "total_correct": 0,
            "error_types": defaultdict(int),
            "missing_by_category": defaultdict(int),
            "extra_by_category": defaultdict(int)
        }

        document_errors = []

        for doc in documents:
            gold_terms = doc.get('gold_terms', [])
            predicted_terms = doc.get('predicted_terms', [])

            # compute number of correct predictions
            gold_set = set(g.lower().strip() for g in gold_terms)
            pred_set = set(p.lower().strip() for p in predicted_terms)
            correct = len(gold_set.intersection(pred_set))
            global_errors["total_correct"] += correct

            # document-level error analysis
            doc_error = self.analyze_document_errors(gold_terms, predicted_terms, doc)

                # tally categories of missing terms
            for missing in doc_error["missing_terms"]:
                category = missing["category"]
                global_errors["missing_by_category"][category] += 1

                for error_type in missing["errors"]:
                    global_errors["error_types"][error_type] += 1

            # tally categories of extra terms
            for extra in doc_error["extra_terms"]:
                category = extra["category"]
                global_errors["extra_by_category"][category] += 1
                global_errors["error_types"][extra["likely_error"]] += 1

            # compute precision, recall, f1 (safe)
            precision = correct / len(predicted_terms) if len(predicted_terms) > 0 else 0
            recall = correct / len(gold_terms) if len(gold_terms) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            document_errors.append({
                "doc_id": doc.get('id', 'unknown'),
                "missing_count": len(doc_error["missing_terms"]),
                "extra_count": len(doc_error["extra_terms"]),
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        # subfield performance analysis
        subfield_analysis = self.analyze_subfield_performance(documents)

        # compute global metrics (safe)
        overall_precision = global_errors["total_correct"] / global_errors["total_predicted_terms"] if global_errors["total_predicted_terms"] > 0 else 0
        overall_recall = global_errors["total_correct"] / global_errors["total_gold_terms"] if global_errors["total_gold_terms"] > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        # generate report
        report = {
            "summary": {
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "overall_f1": overall_f1
            },
            "error_analysis": {
                "error_type_distribution": dict(global_errors["error_types"]),
                "missing_terms_by_category": dict(global_errors["missing_by_category"]),
                "extra_terms_by_category": dict(global_errors["extra_by_category"])
            },
            "subfield_performance": subfield_analysis,
                "document_level_errors": document_errors[:10]  # only keep detailed errors for the first 10 documents
        }

        # save report
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Error analysis report saved to {output_file}")

        return report

    def visualize_error_analysis(self, error_report: Dict):
        """
        Visualize the error analysis results.

        Args:
            error_report: error analysis report dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Error type distribution
        error_types = error_report["error_analysis"]["error_type_distribution"]
        if error_types:
            ax1 = axes[0, 0]
            plt.subplot(2, 2, 1)
            plt.bar(error_types.keys(), error_types.values())
            plt.title("Error Type Distribution")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Frequency")

        # 2. Missing terms by category
        missing_by_category = error_report["error_analysis"]["missing_terms_by_category"]
        if missing_by_category:
            ax2 = axes[0, 1]
            plt.subplot(2, 2, 2)
            plt.bar(missing_by_category.keys(), missing_by_category.values())
            plt.title("Missing Terms by APA Category")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Frequency")

        # 3. Subfield performance comparison
        subfield_data = error_report["subfield_performance"]
        if subfield_data:
            ax3 = axes[1, 0]
            plt.subplot(2, 2, 3)
            subfields = list(subfield_data.keys())
            f1_scores = [subfield_data[sf]["f1"] for sf in subfields]
            plt.bar(subfields, f1_scores)
            plt.title("F1 Score by Subfield")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("F1 Score")

        # 4. Missing terms vs document performance scatter plot
        doc_errors = error_report["document_level_errors"]
        if doc_errors:
            ax4 = axes[1, 1]
            plt.subplot(2, 2, 4)
            missing_counts = [doc["missing_count"] for doc in doc_errors]
            f1_scores = [doc["f1"] for doc in doc_errors]
            plt.scatter(missing_counts, f1_scores)
            plt.title("Missing Terms vs F1 Score")
            plt.xlabel("Number of Missing Terms")
            plt.ylabel("F1 Score")

        plt.tight_layout()
        plt.savefig("error_analysis_visualization.png", dpi=300, bbox_inches='tight')
        print("Error analysis visualization saved to 'error_analysis_visualization.png'")
        plt.show()

def load_extraction_results(results_file: str) -> List[Dict]:
    """
    Load extraction results for error analysis.

    Args:
        results_file: path to the results file

    Returns:
        A list of formatted document dictionaries.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_docs = []
    for i, doc in enumerate(data):
        formatted_docs.append({
            "id": doc.get('id', f"doc_{i}"),
            "title": doc.get('title', ''),
            "abstract": doc.get('abstract', ''),
            "gold_terms": doc.get('llm_annotated_terms', []),
                "predicted_terms": doc.get('system_extracted_terms', [])  # should be extracted from system results
        })

    return formatted_docs

def main():
    """Main entry point: run error analysis."""
    analyzer = TermExtractionErrorAnalyzer()

    # This function expects real extraction results to be available.
    # Example: load a results file and run the analysis.
    try:
        # Load data
        with open("data/abstracts/DEV_set_annotated_LLM_CLEAN.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Simulate predictions (in real use, load system outputs)
        documents = []
        for i, doc in enumerate(data):
            gold_terms = doc.get('llm_annotated_terms', [])
            # In practice, use actual predicted terms; here we simulate partial matches
            predicted_terms = gold_terms[:len(gold_terms)//2] if gold_terms else []

            documents.append({
                "id": doc.get('id', f"doc_{i}"),
                "title": doc.get('title', ''),
                "abstract": doc.get('abstract', ''),
                "gold_terms": gold_terms,
                "predicted_terms": predicted_terms
            })

        # Generate error analysis report
        report = analyzer.generate_error_report(documents, "error_analysis_report.json")

        # Print summary
        print("\n" + "="*50)
        print("ERROR ANALYSIS SUMMARY")
        print("="*50)

        summary = report["summary"]
        print(f"Overall Precision: {summary['overall_precision']:.4f}")
        print(f"Overall Recall: {summary['overall_recall']:.4f}")
        print(f"Overall F1: {summary['overall_f1']:.4f}")

        print(f"\nTop Error Types:")
        error_types = report["error_analysis"]["error_type_distribution"]
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {error_type}: {count}")

        # Visualize results
        analyzer.visualize_error_analysis(report)

    except Exception as e:
        print(f"Error in error analysis: {e}")


if __name__ == "__main__":
    main()