import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import nltk
from typing import List, Set, Dict

# Download stopwords for RAKE
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # Sometimes needed for newer NLTK versions

def calculate_metrics(predicted: List[str], gold: List[str]):
    """
    Calculate Precision, Recall, and F1 based on exact string matching.
    Input terms should be lowercase for comparison.
    """
    pred_set = set(p.lower().strip() for p in predicted)
    gold_set = set(g.lower().strip() for g in gold)
    
    # True Positives: terms in both sets
    true_positives = len(pred_set.intersection(gold_set))
    
    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = true_positives / len(pred_set)
        
    if len(gold_set) == 0:
        recall = 0.0
    else:
        recall = true_positives / len(gold_set)
        
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def run_tfidf_all(documents: List[Dict], top_n=10):
    """
    Runs TF-IDF on the whole corpus, then extracts top N keywords per doc.
    """
    print("Running TF-IDF...")
    # Prepare text corpus
    corpus = [doc.get('abstract', '') + " " + doc.get('title', '') for doc in documents]
    
    # Config: ngram_range=(1, 2) allows capturing phrases like "cognitive dissonance"
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())

    results = []
    
    for i, doc in enumerate(documents):
        # Get row for this document
        row = tfidf_matrix[i].toarray().flatten()
        # Get indices of top N scores
        top_indices = row.argsort()[-top_n:][::-1]
        extracted_keywords = feature_names[top_indices].tolist()
        
        results.append(extracted_keywords)
        
    return results

def run_rake_single(text: str, top_n=10):
    """
    Runs RAKE on a single document text.
    """
    r = Rake(min_length=1, max_length=3) # Allow 1-3 word phrases
    r.extract_keywords_from_text(text)
    # get_ranked_phrases returns list sorted by score
    return r.get_ranked_phrases()[:top_n]

def main():
    # 1. Load Data
    input_file = "../../data/abstracts/DEV_set_annotated_LLM_CLEAN.json"
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Run TF-IDF (Contextual, needs all docs)
    tfidf_predictions = run_tfidf_all(data, top_n=10)

    # 3. Run RAKE and Evaluate
    print("Running RAKE and Evaluating all...")
    
    tfidf_scores = {'p': [], 'r': [], 'f1': []}
    rake_scores = {'p': [], 'r': [], 'f1': []}
    
    for i, doc in enumerate(data):
        text = doc.get('abstract', '') + " " + doc.get('title', '')
        gold_terms = doc.get('llm_annotated_terms', [])
        
        # Skip if no gold terms (avoid division by zero errors in logic)
        if not gold_terms:
            continue

        # --- Evaluate TF-IDF ---
        p, r, f1 = calculate_metrics(tfidf_predictions[i], gold_terms)
        tfidf_scores['p'].append(p)
        tfidf_scores['r'].append(r)
        tfidf_scores['f1'].append(f1)

        # --- Run & Evaluate RAKE ---
        rake_preds = run_rake_single(text, top_n=10)
        p, r, f1 = calculate_metrics(rake_preds, gold_terms)
        rake_scores['p'].append(p)
        rake_scores['r'].append(r)
        rake_scores['f1'].append(f1)

    # 4. Output Average Results
    print("\n" + "="*40)
    print("BASELINE RESULTS (Average on DEV Set)")
    print("="*40)
    
    def print_stats(name, scores):
        print(f"Algorithm: {name}")
        print(f"  Precision: {np.mean(scores['p']):.4f}")
        print(f"  Recall:    {np.mean(scores['r']):.4f}")
        print(f"  F1 Score:  {np.mean(scores['f1']):.4f}")
        print("-" * 20)

    print_stats("TF-IDF", tfidf_scores)
    print_stats("RAKE", rake_scores)

if __name__ == "__main__":
    main()