# Psychology Term Extraction System

基于 APA Thesaurus 的心理学术语提取与概念分析系统

## 项目概述

本项目是 NYU CSCI-UA.0469 Natural Language Processing 课程的期末项目。我们开发了一个针对心理学文本的领域特定术语提取系统，比较了三种提取方法（TF-IDF、RAKE、MentalBERT），并使用 APA Thesaurus 进行概念分析与语义聚类。

## 论文位置

课程论文位于 `acl-style-files-master/acl_latex.tex`，采用 ACL 会议格式撰写。编译方式：

```bash
cd acl-style-files-master
xelatex acl_latex.tex
bibtex acl_latex
xelatex acl_latex.tex
xelatex acl_latex.tex
```

生成的 PDF 为 `acl_latex.pdf`。

## 论文思路

本研究围绕以下核心问题展开：如何从心理学学术文本中自动提取领域特定术语，并将其映射到标准心理学分类体系？

研究路线：

1. 数据构建：从 arXiv 和心理学期刊收集 385 篇论文摘要，从中选取 58 篇进行 LLM 辅助标注，经人工审核后保留 52 篇高质量标注数据（603 个术语实例）

2. 方法对比：实现三种术语提取方法
   - TF-IDF：基于词频统计的传统方法
   - RAKE：基于共现图的关键词提取
   - MentalBERT：基于心理健康领域预训练的 BERT 模型，通过 KeyBERT 框架提取术语

3. 评估体系：采用多维度评估指标
   - 基础指标：Precision、Recall、F1
   - 排序指标：Precision@K
   - 聚类指标：NMI、Cluster Purity、Silhouette Score

4. 概念分析：将提取的术语映射到 APA Thesaurus 的 8 个心理学类别，评估语义一致性

5. 错误分析：按心理学子领域和错误类型进行细粒度分析

## 实验结果

| 方法 | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| TF-IDF | 0.125 | 0.121 | 0.120 |
| RAKE | 0.145 | 0.129 | 0.134 |
| MentalBERT | 0.364 | 0.208 | 0.250 |

MentalBERT 的 F1 分数接近基线方法的两倍，验证了领域适应对术语提取的重要性。

## 文件结构

```
mental/
├── README.md                      # 本文件
├── proposal.md                    # 项目提案
├── acl-style-files-master/        # 论文目录
│   ├── acl_latex.tex              # 论文源文件
│   ├── acl_latex.pdf              # 编译后的论文
│   ├── custom.bib                 # 参考文献
│   └── acl.sty                    # ACL 样式文件
├── data/
│   ├── abstracts/                 # 论文摘要数据
│   │   ├── DEV_set_annotated_LLM_CLEAN.json   # 主评估数据集（52篇）
│   │   ├── DEV_set_annotated_LLM_NOISE.json   # 排除的低质量数据（6篇）
│   │   ├── TEST_set_hidden.json               # 保留的测试集（136篇）
│   │   ├── arxiv_papers.json                  # arXiv 收集数据
│   │   ├── psychology_data_collection.json    # 心理学期刊数据
│   │   └── classic_papers.json                # 经典论文
│   ├── metadata/                  # 数据元信息
│   └── annotation_guidelines.json # 标注指南
├── src/                           # 源代码
│   ├── run_evaluation.py          # 主评估脚本
│   ├── main.py                    # 完整流程入口
│   ├── utils/
│   │   ├── data_collection.py     # 数据爬取
│   │   ├── annotation.py          # LLM 标注
│   │   └── thesaurus.py           # APA Thesaurus 集成
│   ├── evaluation/
│   │   ├── baseline.py            # TF-IDF 和 RAKE 实现
│   │   ├── metrics.py             # 评估指标计算
│   │   └── kappa.py               # Cohen's Kappa
│   ├── models/
│   │   └── mentalbert_extractor.py  # MentalBERT 提取器
│   └── analysis/
│       ├── error_analysis.py      # 错误分析
│       └── semantic.py            # 语义聚类分析
├── output/                        # 实验输出
│   ├── FINAL_COMPREHENSIVE_REPORT.json
│   ├── comprehensive_error_analysis.json
│   ├── comprehensive_semantic_analysis.json
│   ├── comparison_f1.png          # F1 对比图
│   ├── comparison_precision_at_k.png
│   ├── clustering_tsne.png        # t-SNE 聚类可视化
│   ├── apa_category_distribution.png
│   └── cache/                     # 推理缓存
└── outputimg/                     # 其他可视化图表
```

## 运行方式

### 环境配置

```bash
conda create -n mental python=3.10
conda activate mental
pip install numpy pandas scikit-learn nltk rake-nltk keybert sentence-transformers transformers torch matplotlib seaborn tqdm
```

### 运行评估

```bash
cd src
python run_evaluation.py --data ../data/abstracts/DEV_set_annotated_LLM_CLEAN.json
```

可选参数：
- `--no-cache`：禁用缓存
- `--clear-cache`：清除缓存后重新运行

### 数据复现

数据收集：
```python
from utils.data_collection import PsychologyAbstractCollector
collector = PsychologyAbstractCollector("../data")
collector.collect_arxiv_papers(max_papers=50)
```

术语标注：
```python
from utils.annotation import LLMAnnotator
annotator = LLMAnnotator(
    input_file="../data/abstracts/DEV_set_to_annotate.json",
    output_file="../data/abstracts/DEV_set_annotated_LLM.json"
)
annotator.run()
```

## 数据说明

本项目使用 gpt-5-mini 进行初步术语标注，随后进行人工审核。标注遵循以下原则：

提取对象：
- 心理学理论概念（如 cognitive dissonance）
- 心理学构念（如 self-efficacy）
- 实验方法（如 Stroop test）
- 心理现象（如 memory consolidation）

排除对象：
- 通用学术词汇（study, research）
- 非心理学特定的统计术语
- 常见英语词汇

## 参考文献

1. Ji, S., et al. (2022). MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare. LREC 2022.
2. Rose, S., et al. (2010). Automatic Keyword Extraction from Individual Documents. Text Mining: Applications and Theory.
3. American Psychological Association. (2023). APA Thesaurus of Psychological Index Terms (12th ed.).
4. Grootendorst, M. (2020). KeyBERT: Minimal Keyword Extraction with BERT.
