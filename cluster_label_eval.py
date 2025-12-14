"""
File by Jeffrey

Computes accuracy metrics for the clustering compared to the true labels
"""


import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    jaccard_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast


def safe_savefig(path, dpi=600):
    """Only save a figure if the directory exists. Otherwise print a warning."""
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        print(f"[WARNING] Directory does not exist → Skipping save: {path}")
        plt.show()
        return
    plt.savefig(path, dpi=dpi)
    print(f"[saved] {path}")

from enum import Enum

def clustering_evaluation(method, cluster_labels, true_labels, plot_heatmap=True):
    """
    cluster_labels: list (predicted cluster ids; numeric or str)
    true_labels: list (true labels; can be strings)

    ARI, NMI, mapping-based accuracy, per-class metrics, confusion matrix
    mapping based on the Hungarian combinatorial optimization algorithm to maximize correct matches
    """
    figsize = (8, 6)

    y_pred = np.asarray(cluster_labels)
    y_true = np.asarray(true_labels)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Contingency / confusion between predicted clusters and true labels
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    contingency = pd.crosstab(pd.Series(y_true, name="true"),
                              pd.Series(y_pred, name="pred"))

    # Pairwise Jaccard matrix between predicted clusters and true labels
    # jacc_matrix[i, j] = |intersection(true_i, pred_j)| / |union(true_i, pred_j)|
    jacc_matrix = np.zeros((len(true_classes), len(pred_classes)))
    for i, t in enumerate(true_classes):
        idx_t = set(np.where(y_true == t)[0])
        for j, p in enumerate(pred_classes):
            idx_p = set(np.where(y_pred == p)[0])
            inter = len(idx_t & idx_p)
            union = len(idx_t | idx_p)
            jacc_matrix[i, j] = inter / union if union > 0 else 0.0
    jacc_pairwise_df = pd.DataFrame(jacc_matrix,
                                    index=true_classes,
                                    columns=pred_classes)

    # clustering Jaccard metrics (best-match variants)

    # 1) True → best predicted cluster
    # For each true class, take the cluster with highest Jaccard overlap.
    jacc_best_per_true = jacc_pairwise_df.max(axis=1)
    jacc_macro_true_to_pred = float(jacc_best_per_true.mean())

    # 2) Predicted cluster → best true class
    # For each predicted cluster, take the true class with highest Jaccard overlap.
    jacc_best_per_pred = jacc_pairwise_df.max(axis=0)
    jacc_macro_pred_to_true = float(jacc_best_per_pred.mean())

    # Hungarian algo to maximize correct matches:
    # Build cost = -contingency
    cost = -contingency.values
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        true_label = contingency.index[r]
        pred_label = contingency.columns[c]
        mapping[pred_label] = true_label

    # Jaccard restricted to Hungarian-aligned pairs 
    # For each (pred_cluster -> true_class) mapping pair, take its Jaccard
    hungarian_rows = []
    hungarian_jaccs = []
    for pred_label, true_label in mapping.items():
        if (true_label in jacc_pairwise_df.index) and (pred_label in jacc_pairwise_df.columns):
            j = float(jacc_pairwise_df.loc[true_label, pred_label])
            hungarian_rows.append({
                "pred_cluster": pred_label,
                "true_class": true_label,
                "jaccard": j
            })
            hungarian_jaccs.append(j)

    if hungarian_jaccs:
        jacc_macro_hungarian = float(np.mean(hungarian_jaccs))
        hungarian_match_df = pd.DataFrame(hungarian_rows).set_index("pred_cluster")
    else:
        jacc_macro_hungarian = np.nan
        hungarian_match_df = pd.DataFrame(columns=["pred_cluster", "true_class", "jaccard"])

    # Apply mapping to get mapped predicted labels (now in true label space)
    mapped_pred = np.array([mapping.get(p, f"unmapped_{p}") for p in y_pred])
    acc = (mapped_pred == y_true).mean()

    mapped_pred_categories = pd.Categorical(mapped_pred, categories=true_classes)
    cm = pd.crosstab(pd.Series(y_true, name="true"),
                     pd.Series(mapped_pred_categories, name="pred_mapped"))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, mapped_pred, labels=true_classes, zero_division=0
    )

    per_class_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }, index=true_classes)

    print(method)
    print(f"ARI = {ari:.4f}")
    print(f"NMI = {nmi:.4f}")
    print(f"Jaccard macro (true → best cluster) = {jacc_macro_true_to_pred:.4f}")
    print(f"Jaccard macro (cluster → best true) = {jacc_macro_pred_to_true:.4f}")
    print(f"Jaccard macro (Hungarian-aligned pairs) = {jacc_macro_hungarian:.4f}")
    print(f"Mapping-based clustering accuracy = {acc:.4f}")
    print()
    print("Per-class metrics (after mapping):")
    with pd.option_context("display.float_format", "{:0.3f}".format):
        print(per_class_df)
    print()
    print("Contingency table (true x pred):")
    print(contingency)
    print()
    print("Contingency after mapping (true x pred_mapped):")
    print(cm)

    # Best matches (full Jaccard matrix, not restricted to Hungarian)
    print()
    print("Best predicted cluster for each true class (by Jaccard):")
    best_pred_for_true = jacc_pairwise_df.idxmax(axis=1)
    best_jacc_for_true = jacc_pairwise_df.max(axis=1)

    best_match_true_to_pred = pd.DataFrame({
        "best_pred_cluster": best_pred_for_true,
        "jaccard": best_jacc_for_true
    })
    print(best_match_true_to_pred)

    print()
    print("Best true class for each predicted cluster (by Jaccard):")
    best_true_for_pred = jacc_pairwise_df.idxmax(axis=0)
    best_jacc_for_pred = jacc_pairwise_df.max(axis=0)

    best_match_pred_to_true = pd.DataFrame({
        "best_true_class": best_true_for_pred,
        "jaccard": best_jacc_for_pred
    })
    print(best_match_pred_to_true)

    # Hungarian-specific Jaccard table
    print()
    print("Hungarian alignment: Jaccard for each matched (pred_cluster → true_class) pair:")
    print(hungarian_match_df)

    if plot_heatmap:
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Confusion matrix (true labels vs mapped predicted labels)")
        plt.ylabel("True label")
        plt.xlabel("Mapped predicted label")
        plt.tight_layout()
        # plt.savefig(f"./data/confusion_matrix_{method}.png", dpi=600)
        savepath1 = f"./data/confusion_matrix_{method}.png"
        safe_savefig(savepath1)
        plt.close()

        plt.figure(figsize=(max(6, len(pred_classes)*0.5), max(4, len(true_classes)*0.5)))
        sns.heatmap(jacc_pairwise_df, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Pairwise Jaccard (true labels rows × predicted clusters columns)")
        plt.ylabel("True label")
        plt.xlabel("Predicted cluster")
        plt.tight_layout()
        #plt.savefig(f"./data/pairwise_jaccard_{method}.png", dpi=600)
        savepath2 = f"./data/pairwise_jaccard_{method}.png"
        safe_savefig(savepath2) 
        plt.close()

    results = {
        "ARI": ari,
        "NMI": nmi,
        "Jaccard_macro_true_to_pred": jacc_macro_true_to_pred,
        "Jaccard_macro_pred_to_true": jacc_macro_pred_to_true,
        "Jaccard_macro_hungarian": jacc_macro_hungarian,  

        "pairwise_jaccard": jacc_pairwise_df,
        "contingency": contingency,
        "mapping": mapping,
        "mapped_predictions": mapped_pred,
        "mapping_accuracy": acc,
        "confusion_mapped": cm,
        "per_class_metrics": per_class_df,

        "best_match_true_to_pred": best_match_true_to_pred,
        "best_match_pred_to_true": best_match_pred_to_true,
        "hungarian_match_jaccard": hungarian_match_df,    
    }
    return results

class Clustering(Enum):
    Hierarchical = 1
    Spectral = 2
    Kmeans = 3


class Similarity(Enum):
    Cosine = "cosine"
    Euclidean = "euclidean"
    Geodesic = "isomap"

def parse_embeddings_and_type(in_file):
    if 'parquet' in in_file:
        return pd.read_parquet(in_file, columns=["type", "embedding"])

    df = pd.read_csv(
        in_file,
        converters={"embedding": ast.literal_eval},
        usecols=["type", "embedding"],
    )

    return df

def eval_all():
    sim_clust = [
        (Similarity.Cosine, Clustering.Hierarchical),
        (Similarity.Cosine, Clustering.Kmeans),
        (Similarity.Cosine, Clustering.Spectral),
        (Similarity.Euclidean, Clustering.Hierarchical),
        (Similarity.Euclidean, Clustering.Kmeans),
        (Similarity.Euclidean, Clustering.Spectral),
        (Similarity.Geodesic, Clustering.Hierarchical),
        (Similarity.Geodesic, Clustering.Kmeans),
        (Similarity.Geodesic, Clustering.Spectral),
    ]
    '''
    for embedding_name in ['bow', 'esm2_8M', 'esm2_35M', 'esm2_150M', 'prot_bert']:
        embedding_name += '_functional'
        df = parse_embeddings_and_type(f"{embedding_name}.parquet")
    '''

    # structural dataset
    for embedding_name in ['bag_of_words', 'esm2_8M_embeddings', 'esm2_35M_embeddings', 'esm2_150M_embeddings', 'prot_bert_embeddings']:
        #df = parse_embeddings_and_type(f"/home/jc4587/qcb551_proj/embeddings/{embedding_name}.csv")
        df = parse_embeddings_and_type(f"{embedding_name}.csv") 
        # Fix label collumn for esm embeddings
        if 'esm' in embedding_name or 'prot_bert' in embedding_name:
            df['type'] = [s[0] for s in df['type'].tolist()]

        N_PER_CLASS = 200 
        # "Stratified fixed-size sample"
        sampled_df = (
            df.groupby("type", group_keys=False)
            .apply(lambda x: x.sample(n=min(N_PER_CLASS, len(x)), random_state=42))
            .reset_index(drop=True)
        )
        # exclude this category since it only has 23 proteins
        sampled_df = sampled_df[sampled_df['type'] != 'Cell membrane']
        labels = sampled_df['type']

        for similarity, clustering in sim_clust:
            clusters = np.load(f'./data/{embedding_name}_{similarity}_{clustering}_clusterlabels.npy')
            print(embedding_name)
            clustering_evaluation(f'{similarity}_{clustering}', clusters, labels)

# NOTE: run with > 1k_functional.out (for functional dataset) or > 1k_proteins.out (for structural)
if __name__ == "__main__":
    eval_all()

# Example run:

# cluster_labels = [1, 2, 1, 8, 1, 1, 1, 1, 2, 1, 1]
# true_labels    = [
#     "alpha protein", "beta protein", "small protein",
#     "beta protein", "alpha protein", "multidomain protein",
#     "multidomain protein", "alpha protein", "beta protein",
#     "multidomain protein", "alpha protein"
# ]

# res = clustering_evaluation("test_method", cluster_labels, true_labels)


