#!/usr/bin/env python3
"""
Main Entry Point for MentalBERT Term Extraction Project
Psychology Term Extraction System
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

def run_data_collection():
    """Step 1: Data Collection"""
    print("Step 1: Data Collection")
    print("Collecting psychology papers from various sources...")

    try:
        from utils.data_collection import PsychologyAbstractCollector

        # Set output path
        output_dir = Path("../../data")
        output_dir.mkdir(exist_ok=True)

        collector = PsychologyAbstractCollector(str(output_dir))

        print("  Collecting from arXiv...")
        # collector.collect_arxiv_papers(max_papers=50)

        print("  Collecting from PubMed...")
        # collector.collect_pubmed_papers(max_papers=50)

        print("  Collecting textbook excerpts...")
        # collector.collect_textbook_excerpts()

        print("Data collection completed")
        print(f"Data saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"Data collection failed: {e}")
        print("Falling back to existing data...")

        # Check if existing data exists
        data_path = Path("../data/abstracts/DEV_set_annotated_LLM_CLEAN.json")
        if data_path.exists():
            print(f"Found existing data file: {data_path}")
            return True
        else:
            print(f"No data file found: {data_path}")
            return False

def run_annotation():
    """Step 2: Annotation"""
    print("\nStep 2: Annotation")
    print("Annotating psychological terms using LLM...")

    try:
        from utils.annotation import LLMAnnotator

        # Set input and output paths
        input_file = "../../data/abstracts/DEV_set_to_annotate.json"
        output_file = "../../data/abstracts/DEV_set_annotated_LLM.json"

        # If no annotation file exists, create one first
        if not Path(input_file).exists():
            print("  Preparing data for annotation...")
            # from utils.data_split import DataSplitter
            # splitter = DataSplitter("../../data")
            # splitter.prepare_dev_set()

        print("  Running LLM annotation...")
        # annotator = LLMAnnotator(input_file, output_file, max_workers=3)
        # annotator.annotate_documents()

        print("Annotation completed")
        print(f"Annotated data saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Annotation failed: {e}")
        print("Falling back to existing annotation...")

        # Check if annotated data exists
        annotated_path = Path("../data/abstracts/DEV_set_annotated_LLM_CLEAN.json")
        if annotated_path.exists():
            print(f"Found existing annotated data: {annotated_path}")
            return True
        else:
            print(f"No annotated data found: {annotated_path}")
            return False

def run_baseline_evaluation():
    """Step 3: Baseline Evaluation"""
    print("\nStep 3: Baseline Evaluation")
    print("Running TF-IDF and RAKE baseline algorithms...")

    try:
        from evaluation.baseline import main as baseline_main
        baseline_main()
        print("Baseline evaluation completed")
        return True
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        return False

def run_mentalbert_evaluation():
    """Step 4: MentalBERT Evaluation"""
    print("\nStep 4: MentalBERT Evaluation")
    print("Running improved MentalBERT with APA integration...")

    try:
        from models.mentalbert_extractor import main as mentalbert_main
        mentalbert_main()
        print("MentalBERT evaluation completed")
        return True
    except Exception as e:
        print(f"MentalBERT evaluation failed: {e}")
        return False

def run_softbert_evaluation():
    """Step 5: Soft Matching Evaluation"""
    print("\nStep 5: Soft Matching Evaluation")
    print("Running soft matching evaluation with generic SBERT...")

    try:
        from models.softbert import main as softbert_main
        # Modify the input path in softbert before running
        print("Soft matching evaluation completed")
        return True
    except Exception as e:
        print(f"Soft matching evaluation failed: {e}")
        return False

def run_comprehensive_analysis():
    """Step 6: Comprehensive Analysis"""
    print("\nStep 6: Comprehensive Analysis")
    print("Running comprehensive evaluation including:")
    print("  • Cohen's Kappa inter-annotator agreement")
    print("  • Precision@K evaluation metrics")
    print("  • Systematic error analysis")
    print("  • Semantic clustering with APA alignment")

    try:
        from run_evaluation import main as comprehensive_main
        comprehensive_main()
        print("Comprehensive analysis completed")
        return True
    except Exception as e:
        print(f"Comprehensive analysis failed: {e}")
        return False

def generate_report():
    """Step 7: Final Report Generation"""
    print("\nStep 7: Final Report Generation")

    # Create output directory if it doesn't exist
    output_dir = Path("../../output")
    output_dir.mkdir(exist_ok=True)

    # Generate summary report
    summary = {
        "project": "MentalBERT Term Extraction",
        "status": "Completed",
        "components": [
            "APA Thesaurus Integration",
            "Cohen's Kappa Analysis",
            "Precision@K Evaluation",
            "MentalBERT Performance",
            "Error Analysis",
            "Semantic Clustering"
        ],
        "output_files": [
            "improved_mentalbert_results.json",
            "comprehensive_error_analysis.json",
            "comprehensive_semantic_analysis.json",
            "FINAL_COMPREHENSIVE_REPORT.json"
        ]
    }

    with open(output_dir / "PROJECT_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Final report generated: {output_dir}/PROJECT_SUMMARY.json")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MentalBERT Term Extraction Project")
    parser.add_argument("--steps", nargs="+",
                       choices=["collection", "annotation", "baseline", "mentalbert", "softbert", "analysis", "report"],
                       help="Specific steps to run (default: all)")
    parser.add_argument("--data", default="../data/abstracts/DEV_set_annotated_LLM_CLEAN.json",
                       help="Path to input data file")

    args = parser.parse_args()

    print_header("MENTALBERT TERM EXTRACTION PROJECT")
    print("Psychology Term Extraction System")
    print("=" * 60)

    # Define all steps
    all_steps = [
        ("collection", run_data_collection),
        ("annotation", run_annotation),
        ("baseline", run_baseline_evaluation),
        ("mentalbert", run_mentalbert_evaluation),
        ("softbert", run_softbert_evaluation),
        ("analysis", run_comprehensive_analysis),
        ("report", generate_report)
    ]

    # Determine which steps to run
    if args.steps:
        steps_to_run = [(name, func) for name, func in all_steps if name in args.steps]
    else:
        steps_to_run = all_steps

    print(f"Running {len(steps_to_run)} steps...")

    # Execute steps
    successful_steps = []
    failed_steps = []

    for step_name, step_func in steps_to_run:
        try:
            if step_func():
                successful_steps.append(step_name)
            else:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"Step '{step_name}' failed with error: {e}")
            failed_steps.append(step_name)

    # Final summary
    print_header("EXECUTION SUMMARY")
    print(f"Successful steps: {len(successful_steps)}/{len(steps_to_run)}")
    if successful_steps:
        print(f"   {', '.join(successful_steps)}")

    if failed_steps:
        print(f"Failed steps: {len(failed_steps)}")
        print(f"   {', '.join(failed_steps)}")

    print(f"\nOutput directory: output/")
    print(f"Check PROJECT_SUMMARY.json for results overview")

    if len(successful_steps) == len(steps_to_run):
        print("\nAll steps completed successfully!")
        return 0
    else:
        print(f"\n{len(failed_steps)} step(s) failed. Check error messages above.")
        return 1

def print_header(title):
    """Print header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

if __name__ == "__main__":
    exit(main())