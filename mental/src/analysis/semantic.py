"""
Enhanced Semantic Analysis with Clustering Evaluation
Includes cluster purity, NMI evaluation, and APA category alignment analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from typing import List, Dict, Tuple
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.thesaurus import APAThesaurus

class EnhancedSemanticAnalyzer:
    """Enhanced semantic analyzer with clustering evaluation."""

    def __init__(self, model_name='mental/mental-bert-base-uncased'):
        self.model_name = model_name
        self.kw_model = None
        self.apa_thesaurus = APAThesaurus()
        self.term_embeddings = None
        self.term_categories = None

    def load_model(self):
        """Load the embedding/keyword model."""
        if self.kw_model is None:
            print(f"Loading model: {self.model_name}")
            try:
                self.kw_model = KeyBERT(model=self.model_name)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to MiniLM model...")
                self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
                self.model_name = 'all-MiniLM-L6-v2'

    def extract_and_embed_terms(self, documents: List[Dict], max_docs: int = 50) -> Tuple[List[str], np.ndarray]:
        """
        Extract terms from documents and generate embeddings.

        Args:
            documents: list of document dicts
            max_docs: maximum number of documents to process

        Returns:
            (unique_terms_list, corresponding_embedding_matrix)
        """
        if self.kw_model is None:
            self.load_model()

        print(f"Extracting terms from {min(len(documents), max_docs)} documents...")
        collected_terms = set()
        term_sources = {}  # record source documents for each term

        for doc in documents[:max_docs]:
            text = doc.get('title', '') + ". " + doc.get('abstract', '')
            gold_terms = doc.get('llm_annotated_terms', [])

            # Prefer gold-standard terms when available
            terms_to_use = gold_terms if gold_terms else []

            # If gold terms are insufficient, extract additional keywords
            if len(terms_to_use) < 3:
                try:
                    keywords = self.kw_model.extract_keywords(
                        text,
                        keyphrase_ngram_range=(1, 2),
                        stop_words='english',
                        top_n=5
                    )
                    extracted_terms = [kw[0] for kw in keywords]
                    terms_to_use.extend(extracted_terms)
                except Exception as e:
                    print(f"Error extracting keywords: {e}")

            # Collect terms
            for term in terms_to_use:
                term_clean = term.lower().strip()
                if len(term_clean.split()) <= 3:  # limit term length
                    if term_clean not in collected_terms:
                        term_sources[term_clean] = []
                    term_sources[term_clean].append(doc.get('id', 'unknown'))
                    collected_terms.add(term_clean)

        unique_terms = list(collected_terms)
        print(f"Found {len(unique_terms)} unique terms")

        # Generate embeddings
        print("Generating embeddings...")
        try:
            # Use KeyBERT's internal model to generate embeddings
            embeddings = self.kw_model.model.embed(unique_terms)
            self.term_embeddings = embeddings
            self.term_categories = [self.apa_thesaurus.categorize_term(term) for term in unique_terms]
            return unique_terms, embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # return random embeddings as a fallback
            embeddings = np.random.rand(len(unique_terms), 384)
            return unique_terms, embeddings

    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int = 8) -> np.ndarray:
        """
        Perform clustering on embeddings.

        Args:
            embeddings: embedding matrix
            n_clusters: number of clusters

        Returns:
            cluster labels array
        """
        print(f"Performing clustering with {n_clusters} clusters...")

        # multiple clustering methods
        clustering_methods = {
            "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        }

        cluster_results = {}
        for method_name, clustering_method in clustering_methods.items():
            try:
                labels = clustering_method.fit_predict(embeddings)
                cluster_results[method_name] = labels
                print(f"{method_name} clustering completed")
            except Exception as e:
                print(f"{method_name} clustering failed: {e}")

        # return the best available result (prefer KMeans)
        return cluster_results.get("KMeans", cluster_results.get("Agglomerative", np.zeros(len(embeddings))))

    def calculate_clustering_metrics(self, embeddings: np.ndarray,
                                   cluster_labels: np.ndarray,
                                   true_labels: List[str]) -> Dict:
        """
        Calculate clustering evaluation metrics.

        Args:
            embeddings: embedding matrix
            cluster_labels: cluster labels
            true_labels: ground-truth labels

        Returns:
            dictionary of clustering metrics
        """
        metrics = {}

        # 1. NMI (Normalized Mutual Information)
        try:
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            metrics['nmi'] = nmi
        except Exception as e:
            print(f"Error calculating NMI: {e}")
            metrics['nmi'] = 0.0

        # 2. ARI (Adjusted Rand Index)
        try:
            ari = adjusted_rand_score(true_labels, cluster_labels)
            metrics['ari'] = ari
        except Exception as e:
            print(f"Error calculating ARI: {e}")
            metrics['ari'] = 0.0

        # 3. Silhouette Score
        try:
            if len(set(cluster_labels)) > 1:  # ensure multiple clusters exist
                silhouette = silhouette_score(embeddings, cluster_labels, metric='cosine')
                metrics['silhouette'] = silhouette
            else:
                metrics['silhouette'] = 0.0
        except Exception as e:
            print(f"Error calculating Silhouette Score: {e}")
            metrics['silhouette'] = 0.0

        # 4. Cluster Purity
        try:
            purity = self._calculate_cluster_purity(cluster_labels, true_labels)
            metrics['purity'] = purity
        except Exception as e:
            print(f"Error calculating Purity: {e}")
            metrics['purity'] = 0.0

        return metrics

    def _calculate_cluster_purity(self, cluster_labels: np.ndarray, true_labels: List[str]) -> float:
        """
        Calculate cluster purity.

        Args:
            cluster_labels: cluster labels
            true_labels: ground-truth labels

        Returns:
            purity score (float)
        """
        # build contingency matrix
        contingency = contingency_matrix(true_labels, cluster_labels)

        # compute purity
        purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
        return purity

    def analyze_apa_category_alignment(self, terms: List[str],
                                     cluster_labels: np.ndarray) -> Dict:
        """
        Analyze alignment between clusters and APA categories.

        Args:
            terms: list of terms
            cluster_labels: cluster labels

        Returns:
            APA category alignment analysis
        """
        # get APA category for each term
        apa_categories = [self.apa_thesaurus.categorize_term(term) for term in terms]

        # Analyze dominant APA categories per cluster
        cluster_analysis = {}
        unique_clusters = set(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_apa_categories = [apa_categories[i] for i in cluster_indices]

            # count APA category distribution
            category_counts = {}
            for category in cluster_apa_categories:
                category_counts[category] = category_counts.get(category, 0) + 1

            # find the dominant category
            dominant_category = max(category_counts, key=category_counts.get)
            dominance_ratio = category_counts[dominant_category] / len(cluster_apa_categories)

            cluster_analysis[cluster_id] = {
                "size": len(cluster_indices),
                "dominant_apa_category": dominant_category,
                "dominance_ratio": dominance_ratio,
                "category_distribution": category_counts,
                "sample_terms": [terms[i] for i in cluster_indices[:5]]  # sample terms (up to 5)
            }

        # compute overall alignment quality
        total_terms = len(terms)
        well_aligned_terms = sum(
            analysis["size"] * analysis["dominance_ratio"]
            for analysis in cluster_analysis.values()
        )
        overall_alignment = well_aligned_terms / total_terms if total_terms > 0 else 0

        return {
            "cluster_analysis": cluster_analysis,
            "overall_alignment_score": overall_alignment,
            "num_clusters": len(unique_clusters),
            "unique_apa_categories": len(set(apa_categories))
        }

    def visualize_clustering_results(self, terms: List[str],
                                   embeddings: np.ndarray,
                                   cluster_labels: np.ndarray,
                                   apa_categories: List[str],
                                   save_dir: str = "."):
        """
        Visualize clustering results - generate three separate figures.

        Args:
            terms: list of terms
            embeddings: embedding matrix
            cluster_labels: cluster labels
            apa_categories: APA category labels
            save_dir: directory to save images
        """
        print("Creating clustering visualizations...")
        
        from matplotlib.lines import Line2D
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(terms)-1))
        coords_2d = tsne.fit_transform(embeddings)

        # ========== Figure 1: Cluster scatter plot ==========
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        scatter1 = ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                               c=cluster_labels, cmap='tab10', s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax1.set_title("Term Clusters (t-SNE Visualization)", fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel("Semantic Dimension 1", fontsize=12)
        ax1.set_ylabel("Semantic Dimension 2", fontsize=12)
        
        # add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label("Cluster ID", fontsize=11)
        
        # annotate a few representative terms (up to 2 per cluster)
        np.random.seed(42)
        for cluster_id in set(cluster_labels):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            sample_indices = np.random.choice(cluster_indices, min(2, len(cluster_indices)), replace=False)
            for idx in sample_indices:
                ax1.annotate(terms[idx], (coords_2d[idx, 0], coords_2d[idx, 1]),
                           fontsize=9, alpha=0.85, xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        save_path1 = os.path.join(save_dir, "clustering_tsne.png")
        plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path1}")
        plt.close()

        # ========== Figure 2: APA category distribution scatter plot ==========
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        unique_categories = list(set(apa_categories))
        category_colors = {cat: i for i, cat in enumerate(unique_categories)}
        category_colors_list = [category_colors[cat] for cat in apa_categories]
        
        cmap = plt.cm.Set2
        scatter2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                               c=category_colors_list, cmap='Set2', s=80, alpha=0.7, 
                               edgecolors='white', linewidth=0.5)
        ax2.set_title("APA Thesaurus Category Distribution", fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel("Semantic Dimension 1", fontsize=12)
        ax2.set_ylabel("Semantic Dimension 2", fontsize=12)
        
        # create legend
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=cmap(category_colors[cat] / max(len(unique_categories)-1, 1)), 
                                  markersize=12, label=cat, markeredgecolor='gray', markeredgewidth=0.5)
                          for cat in unique_categories]
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
                   fontsize=10, title="APA Categories", title_fontsize=11)
        
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        save_path2 = os.path.join(save_dir, "apa_category_distribution.png")
        plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path2}")
        plt.close()

        # ========== Figure 3: Cluster-APA category confusion heatmap ==========
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        try:
            # create contingency matrix
            conf_matrix = contingency_matrix(apa_categories, cluster_labels)
            
            # normalize contingency matrix (by row)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix_norm = np.divide(conf_matrix.astype('float'), row_sums, 
                                         where=row_sums!=0, out=np.zeros_like(conf_matrix, dtype=float))
            
            # plot heatmap
            cluster_ids = sorted(set(cluster_labels))
            category_names = unique_categories
            
            im = ax3.imshow(conf_matrix_norm, cmap='YlOrRd', aspect='auto')
            ax3.set_title("Cluster-APA Category Alignment Heatmap", fontsize=16, fontweight='bold', pad=15)
            ax3.set_xlabel("Cluster ID", fontsize=12)
            ax3.set_ylabel("APA Category", fontsize=12)
            
            # add colorbar
            cbar3 = plt.colorbar(im, ax=ax3)
            cbar3.set_label("Normalized Proportion", fontsize=11)
            
            # set tick labels
            ax3.set_xticks(range(len(cluster_ids)))
            ax3.set_xticklabels([f"C{i}" for i in cluster_ids], fontsize=10)
            ax3.set_yticks(range(len(category_names)))
            ax3.set_yticklabels(category_names, fontsize=10)
            
            # annotate each cell with its value
            for i in range(len(category_names)):
                for j in range(len(cluster_ids)):
                    value = conf_matrix_norm[i, j]
                    text_color = 'white' if value > 0.5 else 'black'
                    ax3.text(j, i, f'{value:.2f}', ha='center', va='center', 
                            color=text_color, fontsize=9, fontweight='bold')

        except Exception as e:
            ax3.text(0.5, 0.5, f"Heatmap Error:\n{str(e)}",
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        plt.tight_layout()
        save_path3 = os.path.join(save_dir, "cluster_apa_heatmap.png")
        plt.savefig(save_path3, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path3}")
        plt.close()
        
        print(f"All 3 clustering visualizations saved to '{save_dir}'")

    def run_comprehensive_analysis(self, documents: List[Dict],
                                 output_file: str = "comprehensive_semantic_analysis.json",
                                 output_dir: str = "."):
        """
        Run comprehensive semantic analysis.

        Args:
            documents: list of document dicts
            output_file: path to save the output JSON
            output_dir: output directory for visualization images

        Returns:
            the full analysis results as a dict
        """
        print("Starting comprehensive semantic analysis...")

        # 1. extract terms and generate embeddings
        terms, embeddings = self.extract_and_embed_terms(documents)

        # 2. perform clustering
        n_clusters = min(8, len(terms) // 5)  # determine number of clusters dynamically
        cluster_labels = self.perform_clustering(embeddings, n_clusters)

        # 3. get APA categories
        apa_categories = [self.apa_thesaurus.categorize_term(term) for term in terms]

        # 4. compute clustering metrics
        clustering_metrics = self.calculate_clustering_metrics(embeddings, cluster_labels, apa_categories)

        # 5. analyze APA category alignment
        apa_alignment = self.analyze_apa_category_alignment(terms, cluster_labels)

        # 6. generate visualizations (three separate figures)
        self.visualize_clustering_results(terms, embeddings, cluster_labels, apa_categories, save_dir=output_dir)

        
        def convert_to_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        analysis_results = {
            "summary": {
                "total_terms_analyzed": len(terms),
                "embedding_model": self.model_name,
                "clustering_algorithm": "KMeans",
                "num_clusters": n_clusters,
                "num_apa_categories": len(set(apa_categories))
            },
            "clustering_metrics": convert_to_serializable(clustering_metrics),
            "apa_category_alignment": convert_to_serializable(apa_alignment),
            "term_data": {
                "terms": terms,
                "cluster_labels": cluster_labels.tolist(),
                "apa_categories": apa_categories
            },
            "cluster_details": {}
        }

        # add detailed info for each cluster
        for cluster_id, analysis in apa_alignment["cluster_analysis"].items():
            analysis_results["cluster_details"][f"cluster_{cluster_id}"] = convert_to_serializable(analysis)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        print(f"Comprehensive analysis saved to '{output_file}'")

        # print summary
        self.print_analysis_summary(analysis_results)

        return analysis_results

    def print_analysis_summary(self, results: Dict):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("COMPREHENSIVE SEMANTIC ANALYSIS SUMMARY")
        print("="*60)

        summary = results["summary"]
        metrics = results["clustering_metrics"]
        alignment = results["apa_category_alignment"]

        print(f"Dataset Overview:")
        print(f"  Terms Analyzed: {summary['total_terms_analyzed']}")
        print(f"  Embedding Model: {summary['embedding_model']}")
        print(f"  Number of Clusters: {summary['num_clusters']}")
        print(f"  APA Categories: {summary['num_apa_categories']}")

        print(f"\nClustering Quality:")
        print(f"  NMI Score: {metrics['nmi']:.3f}")
        print(f"  ARI Score: {metrics['ari']:.3f}")
        print(f"  Silhouette Score: {metrics['silhouette']:.3f}")
        print(f"  Cluster Purity: {metrics['purity']:.3f}")

        print(f"\nAPA Category Alignment:")
        print(f"  Overall Alignment: {alignment['overall_alignment_score']:.3f}")
        print(f"  Well-aligned Clusters: {sum(1 for c in alignment['cluster_analysis'].values() if c['dominance_ratio'] > 0.7)}/{alignment['num_clusters']}")

        print(f"\nTop Performing Clusters:")
        sorted_clusters = sorted(
            alignment['cluster_analysis'].items(),
            key=lambda x: x[1]['dominance_ratio'],
            reverse=True
        )[:3]

        for cluster_id, analysis in sorted_clusters:
            print(f"  Cluster {cluster_id}: {analysis['dominant_apa_category']} (purity: {analysis['dominance_ratio']:.3f})")

        print("-" * 60)

def main():
    """Main entry: run comprehensive semantic analysis."""
    # load data
    with open("data/abstracts/DEV_set_annotated_LLM_CLEAN.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # create analyzer and run analysis
    analyzer = EnhancedSemanticAnalyzer()
    results = analyzer.run_comprehensive_analysis(data, "comprehensive_semantic_analysis.json")

if __name__ == "__main__":
    main()