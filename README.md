# Psychology Term Extraction System

A system for domain-specific psychological term extraction and conceptual analysis based on the APA Thesaurus.

## Project Overview

This project is the final project for NYU CSCI-UA.0469 Natural Language Processing (Spring 2024). We developed a domain-adapted term extraction system for psychological academic texts, compared three extraction methods (TF-IDF, RAKE, MentalBERT), and performed conceptual alignment and semantic clustering using the APA Thesaurus taxonomy.。

## Paper Location

The course paper is located at acl-style-files-master/acl_latex.tex, written in ACL conference format. Compilation steps:

```bash
cd acl-style-files-master
xelatex acl_latex.tex
bibtex acl_latex
xelatex acl_latex.tex
xelatex acl_latex.tex
```

The generated PDF is acl_latex.pdf

## Research Methodology

This study focuses on the core research question: How to automatically extract domain-specific psychological terms from academic abstracts and map them into a standardized psychological taxonomy?

Research pipeline:

1. Data Construction: Collected 385 psychology paper abstracts from arXiv and major psychology journals. 58 were selected for LLM-assisted annotation → 52 retained after human validation, producing 603 high-quality psychological term instances

2. Method Comparison: Implemented three term extraction approaches:

   TF-IDF: A traditional frequency-statistics baseline

   RAKE: Graph-based co-occurrence keyword extraction

   MentalBERT: A BERT model pretrained in the mental-health domain, extracting terms through the KeyBERT framework

3. Evaluation Framework: Adopted multi-dimensional evaluation metrics:

   Base metrics: Precision, Recall, F1

   Ranking metrics: Precision@K

   Clustering metrics: NMI, Cluster Purity, Silhouette Score

4. Concept Mapping & Analysis: Mapped extracted terms into the 8 psychological categories defined in the APA Thesaurus,       evaluating semantic consistency and taxonomy alignment

5. Error Analysis: Conducted fine-grained analysis by psychology sub-domain and extraction failure mode

Main Results

| Method| Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| TF-IDF | 0.125 | 0.121 | 0.120 |
| RAKE | 0.145 | 0.129 | 0.134 |
| MentalBERT | 0.364 | 0.208 | 0.250 |

MentalBERT achieves nearly 2× F1 compared to baseline methods, verifying the importance of domain adaptation for psychological term extraction.

## File Structure

```
mental/
├── README.md                      # This file
├── proposal.md                    # Project proposal
├── acl-style-files-master/        # Paper directory
│   ├── acl_latex.tex              # Paper source
│   ├── acl_latex.pdf              # Compiled paper
│   ├── custom.bib                 # References
│   └── acl.sty                    # ACL style file
├── data/
│   ├── abstracts/                 # Abstract datasets
│   │   ├── DEV_set_annotated_LLM_CLEAN.json   # Main evaluation set (52 abstracts)
│   │   ├── DEV_set_annotated_LLM_NOISE.json   # Excluded low-quality data (6 abstracts)
│   │   ├── TEST_set_hidden.json               # Held-out test set (136 abstracts)
│   │   ├── arxiv_papers.json                  # Raw arXiv abstract collection
│   │   ├── psychology_data_collection.json    # Collected psychology journal data
│   │   └── classic_papers.json                # Classic psychology abstract set
│   ├── metadata/                  # Dataset metadata
│   └── annotation_guidelines.json # Annotation guideline file
├── src/                           # Source code
│   ├── run_evaluation.py          # Main evaluation script
│   ├── main.py                    # End-to-end pipeline entry
│   ├── utils/
│   │   ├── data_collection.py     # Data crawling & preprocessing
│   │   ├── annotation.py          # LLM annotation module
│   │   └── thesaurus.py           # APA Thesaurus integration
│   ├── evaluation/
│   │   ├── baseline.py            # TF-IDF and RAKE implementation
│   │   ├── metrics.py             # Metric computation file
│   │   └── kappa.py               # Cohen’s Kappa implementation
│   ├── models/
│   │   └── mentalbert_extractor.py  # MentalBERT extractor
│   └── analysis/
│       ├── error_analysis.py      # Error analysis module
│       └── semantic.py            # Semantic clustering module
├── output/
│   ├── FINAL_COMPREHENSIVE_REPORT.json
│   ├── comprehensive_error_analysis.json
│   ├── comprehensive_semantic_analysis.json
│   ├── comparison_f1.png          # F1 comparison chart
│   ├── comparison_precision_at_k.png
│   ├── clustering_tsne.png        # t-SNE clustering visualization
│   ├── apa_category_distribution.png
│   └── cache/                     # Inference cache directory
└── outputimg/                     # Additional visualization charts

```

## Run Instructions

### Environment Setup

```bash
conda create -n mental python=3.10
conda activate mental
pip install numpy pandas scikit-learn nltk rake-nltk keybert sentence-transformers transformers torch matplotlib seaborn tqdm
```

### Run Evaluation

```bash
cd src
python run_evaluation.py --data ../data/abstracts/DEV_set_annotated_LLM_CLEAN.json
```

Optional parameters:
- `--no-cache`：disable inference cache
- `--clear-cache`：clear cache before re-running

### Reproduce Data & Annotation

Collect new abstracts:
```python
from utils.data_collection import PsychologyAbstractCollector
collector = PsychologyAbstractCollector("../data")
collector.collect_arxiv_papers(max_papers=50)
```

Annotate psychological terms:
```python
from utils.annotation import LLMAnnotator
annotator = LLMAnnotator(
    input_file="../data/abstracts/DEV_set_to_annotate.json",
    output_file="../data/abstracts/DEV_set_annotated_LLM.json"
)
annotator.run()
```

### Data Description

This project uses GPT-5-mini for initial psychological term annotation, followed by manual human review. The annotation process follows these domain-specific principles:

Included term types:
- Psychological theories (e.g., cognitive dissonance)

- Psychological constructs (e.g., self-efficacy)

- Experimental paradigms (e.g., Stroop test)

- Psychological phenomena (e.g., memory consolidation)

Excluded term types:

- Generic academic words (study, research, analysis)

- Non-psychology-specific statistical terminology

- Common English vocabulary

## References

1. Ji, S., et al. (2022). MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare. LREC 2022.
2. Rose, S., et al. (2010). Automatic Keyword Extraction from Individual Documents. Text Mining: Applications and Theory.
3. American Psychological Association. (2023). APA Thesaurus of Psychological Index Terms (12th ed.).
4. Grootendorst, M. (2020). KeyBERT: Minimal Keyword Extraction with BERT.
