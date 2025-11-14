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

def clustering_evaluation(method, cluster_labels, true_labels, plot_heatmap=True):
    """
    cluster_labels: list (predicted cluster ids; numeric or str)
    true_labels: list (true labels; can be strings)

    ARI, NMI, mapping-based accuracy, per-class metrics, confusion matrix
    mapping based on the Hungarian combinatorial optimization algorithm to maximize correct matches
    """
    figsize=(8,6) 

    y_pred = np.asarray(cluster_labels)
    y_true = np.asarray(true_labels)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Contingency / confusion between predicted clusters and true labels
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    contingency = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))

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
    jacc_pairwise_df = pd.DataFrame(jacc_matrix, index=true_classes, columns=pred_classes)

    # clustering Jaccard metrics (best-match variants)
    # For each true class, best overlapping predicted cluster (true -> best pred)
    # then averages those best overlaps.
    jacc_best_per_true = jacc_pairwise_df.max(axis=1)       
    jacc_macro = float(jacc_best_per_true.mean())            

    # Hungarian algo to maximize correct matches:
    # Build cost = -contingency 
    cost = -contingency.values
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        true_label = contingency.index[r]
        pred_label = contingency.columns[c]
        mapping[pred_label] = true_label

    mapped_pred = np.array([mapping.get(p, f"unmapped_{p}") for p in y_pred])
    acc = (mapped_pred == y_true).mean()

    mapped_pred_categories = pd.Categorical(mapped_pred, categories=true_classes)
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(mapped_pred_categories, name="pred_mapped"))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, mapped_pred, labels=true_classes, zero_division=0)

    per_class_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }, index=true_classes)

    print(method)
    print(f"ARI = {ari:.4f}")
    print(f"NMI = {nmi:.4f}")
    print(f"Jaccard (macro) = {jacc_macro:.4f}")
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

    best_pred_for_true = jacc_pairwise_df.idxmax(axis=1)
    best_jacc_for_true = jacc_pairwise_df.max(axis=1)
    best_match_df = pd.DataFrame({"best_pred_cluster": best_pred_for_true, "jaccard": best_jacc_for_true})
    print()
    print("Best predicted cluster for each true class (by Jaccard):")
    print(best_match_df)

    if plot_heatmap:
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Confusion matrix (true labels vs mapped predicted labels)")
        plt.ylabel("True label")
        plt.xlabel("Mapped predicted label")
        plt.tight_layout()
        plt.savefig(f"./data/confusion_matrix_{method}.png", dpi=600)
        plt.close()

        plt.figure(figsize=(max(6, len(pred_classes)*0.5), max(4, len(true_classes)*0.5)))
        sns.heatmap(jacc_pairwise_df, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Pairwise Jaccard (true labels rows × predicted clusters columns)")
        plt.ylabel("True label")
        plt.xlabel("Predicted cluster")
        plt.tight_layout()
        plt.savefig(f"./data/pairwise_jaccard_{method}.png", dpi=600)
        plt.close()

    results = {
        "ARI": ari,
        "NMI": nmi,
        "Jaccard_macro": jacc_macro,
        "pairwise_jaccard": jacc_pairwise_df,
        "contingency": contingency,
        "mapping": mapping,
        "mapped_predictions": mapped_pred,
        "mapping_accuracy": acc,
        "confusion_mapped": cm,
        "per_class_metrics": per_class_df,
        "best_match_df": best_match_df,
    }
    return results

# Example run:

# cluster_labels = [1, 2, 1, 8, 1, 1, 1, 1, 2, 1, 1]
# true_labels    = [
#     "alpha protein", "beta protein", "small protein",
#     "beta protein", "alpha protein", "multidomain protein",
#     "multidomain protein", "alpha protein", "beta protein",
#     "multidomain protein", "alpha protein"
# ]

# res = clustering_evaluation(cluster_labels, true_labels) 
