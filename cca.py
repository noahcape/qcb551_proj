from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
import matplotlib.patches as mpatches
import pandas as pd
import pickle
import os 


def apply_pca_to_embeddings(embeddings_dict, n_components=10):
    """
    embeddings_dict: dict of {name: embedding (np.array)}
    """
    pca_embeddings = {}
    for name, embedding in embeddings_dict.items():
        pca = PCA(n_components=n_components)
        embedding_reduced = pca.fit_transform(embedding)
        pca_embeddings[name] = embedding_reduced
    return pca_embeddings


def compute_similarity(emb1, emb2):
    """
    Compute CCA similarity between two embeddings.
    how much two different representations of the same entities capture the same underlying structure
    """
    n_components = min(emb1.shape[1], emb2.shape[1])
    cca = CCA(n_components=n_components)
    cca.fit(emb1, emb2)
    # X_c, Y_c = cca.transform(emb1, emb2)
    return cca.score(emb1, emb2)


if __name__ == "__main__":

    # need to modify once we have the embeddings ... 
    # but generally create a dict of embedding name and the embedding itself
    EMBEDDING_PATH = "embeddings/"
    with open(os.path.join(EMBEDDING_PATH, "embeddings_dict.pkl"), "rb") as f:
        embeddings_aligned = pickle.load(f)

    embeddings_np = {
        name: np.array(embedding) for name, embedding in embeddings_aligned.items()
    }
    embeddings_np_pca = apply_pca_to_embeddings(
        embeddings_np, n_components=10
    )

    embedding_names = sorted(embeddings_np_pca.keys())
    n_embeddings = len(embedding_names)
    similarity_matrix = np.zeros((n_embeddings, n_embeddings))

    for i, name_i in enumerate(embedding_names):
        print(name_i)
        for j, name_j in enumerate(embedding_names[i:], start=i):
            sim = compute_similarity(
                embeddings_np_pca[name_i], embeddings_np_pca[name_j]
            )
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    sim_df = pd.DataFrame(similarity_matrix, index=embedding_names, columns=embedding_names)


    g = sns.clustermap(
        sim_df,
        annot=False,
        cmap="vlag",
        fmt=".4f",
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.5},
        figsize=(20, 20),
    )

    g.fig.suptitle("Canonical Correlation Analysis Scores", fontsize=16)

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right", fontsize=20)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=20)


    plt.savefig("cca_heatmap.pdf", format="pdf", bbox_inches="tight")

    plt.show()


## i also have code for pearson correlation between embeddings, but it's more convoluted...
## cca is easier to interpret... 
def embd_weighted_pearson_jaccard(embeddings_dict):
    """
    Compute a weighted Jaccard similarity between protein protein correlation
    structures across multiple embeddings.
    returns square similarity matrix (names x names) with weighted Jaccard scores or correlations.
    """

    dfs = {k: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)) for k, v in embeddings_dict.items()}
    protein_sets = [set(df.columns) for df in dfs.values()]
    common_proteins = set.intersection(*protein_sets) if protein_sets else set()

    if len(common_proteins) < 2:
        raise ValueError("Need at least two common proteins across all embeddings to compute correlations.")

    # Consistent column order
    proteins = sorted(common_proteins)

    # Compute scaled (0..1) upper-tri correlation vectors per embedding
    scaled_vectors = {}
    for name, df in dfs.items():
        sub = df.loc[:, proteins]
        # protein–protein corr (columns vs columns) using Pearson; fill any NaNs with 0
        prot_corr = sub.corr(method="pearson").fillna(0.0).to_numpy()
        # scale to [0,1]
        prot_corr_scaled = (prot_corr + 1.0) / 2.0
        # take upper triangle (k=1 to exclude diagonal)
        tri = np.triu_indices(len(proteins), k=1)
        scaled_vectors[name] = prot_corr_scaled[tri]

    names = sorted(scaled_vectors.keys())
    n = len(names)
    sim_mat = np.zeros((n, n), dtype=float)

    # Weighted Jaccard between vectors
    for i in range(n):
        v1 = scaled_vectors[names[i]]
        for j in range(i, n):
            v2 = scaled_vectors[names[j]]
            if v1.shape != v2.shape:
                raise ValueError(f"Vector length mismatch between {names[i]} and {names[j]}")
            min_sum = np.minimum(v1, v2).sum()
            max_sum = np.maximum(v1, v2).sum()
            wj = (min_sum / max_sum) if max_sum > 0 else 0.0
            sim_mat[i, j] = sim_mat[j, i] = wj

    return pd.DataFrame(sim_mat, index=names, columns=names)