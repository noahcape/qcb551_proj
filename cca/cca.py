from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
import matplotlib.patches as mpatches
import pandas as pd
import pickle
import os

from sklearn.discriminant_analysis import StandardScaler 


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


def compute_similarity(emb1, emb2, n_components=None, score = False):
    """
    Compute CCA similarity between two embeddings.
    how much two different representations of the same entities capture the same underlying structure
    """
    if n_components is None:
        n_components = min(emb1.shape[1], emb2.shape[1])

    X = StandardScaler().fit_transform(emb1)
    Y = StandardScaler().fit_transform(emb2)
    
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)

    if score:
        return cca.score(X, Y)
    
    X_c, Y_c = cca.transform(X, Y)
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
    corrs = np.clip(np.abs(corrs), 0.0, 1.0)
    return float(np.mean(corrs))


def embd_weighted_pearson_jaccard(embeddings_dict):
    """
    an alternative to CCA if we want, prob don't need this function 
    
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



def compute_similarity_matrix(embeddings_dict, n_pca_components = 10, cca_score = False):
    """
    embeddings_dict: dict of {name: 2D np.array (n_entities x embedding_dim)}
    """
    lengths = {name: arr.shape[0] for name, arr in embeddings_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Embeddings have different numbers of entities: {lengths}")

    pca_embs = apply_pca_to_embeddings(embeddings_dict, n_components=n_pca_components)

    names = sorted(pca_embs.keys())
    n = len(names)
    sim_mat = np.zeros((n, n), dtype=float)

    for i, ni in enumerate(names):
        Xi = pca_embs[ni]
        for j, nj in enumerate(names[i:], start=i):
            Xj = pca_embs[nj]
            sim = compute_similarity(Xi, Xj, n_components=None, score=cca_score)
            sim_mat[i, j] = sim_mat[j, i] = sim

    sim_df = pd.DataFrame(sim_mat, index=names, columns=names)
    return sim_df