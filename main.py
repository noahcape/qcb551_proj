from enum import Enum
import scipy as sc
import sklearn as sk
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap

SCOPe_fname = "astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa" #"./astral-scopedom-seqres-gd-all-2.08-stable.fa"


class Clustering(Enum):
    Hierarchical = 1
    Spectral = 2
    Kmeans = 3


class Similarity(Enum):
    Cosine = "cosine"
    Euclidean = "euclidean"
    Geodesic = "isomap"


def create_similarity_matrix(similarity, data):
    if similarity == Similarity.Geodesic:
        isomap = sk.manifold.Isomap(n_neighbors=3, n_components=2)
        isomap.fit_transform(data)
        return isomap.dist_matrix_
    else:
        return sc.spatial.distance.squareform(
            sc.spatial.distance.pdist(data, metric=similarity.value)
        )


def cluster(similarity_m, clustering, n):
    match clustering:
        case Clustering.Spectral:
            spectral = sk.cluster.SpectralClustering(
                n, affinity="precomputed", n_init=100, assign_labels="discretize"
            )
            return spectral.fit_predict(similarity_m)
        case Clustering.Hierarchical:
            clusters = sk.cluster.AgglomerativeClustering(linkage="average").fit(
                similarity_m
            )
            return clusters.labels_
        case Clustering.Kmeans:
            return sk.cluster.KMeans(
                n_clusters=n, random_state=0, n_init="auto"
            ).fit_predict(similarity_m)


aa_ordering = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

aa_indices = {a: i for (a, i) in list(zip(aa_ordering, range(20)))}


def bow_embedding(p):
    embedding = [0 for _ in range(20)]
    for aa in p:
        # this is handeling the occurance of X
        if aa not in aa_ordering:
            continue
        embedding[aa_indices[aa]] += 1

    return embedding


def parse_SCOPe_file(embedding_func, outname, verbose=True):
    pos = 0
    with open(f"./{outname}", "w") as out_file:
        out_file.write("name,type,sequence,embedding\n")
        with open(SCOPe_fname, "r") as in_file:
            p = ""
            name = None
            s_type = None
            for line in in_file:
                line = line.strip()
                if line.startswith(">"):
                    pos += 1
                    if verbose and pos % 1000 == 0:
                        print(pos)
                    # If we already have a previous sequence, process it
                    if p and name and s_type:
                        embedding = embedding_func(p.upper())
                        out_file.write(f'{name},{s_type},{p},"{embedding}"\n')

                    # Start a new sequence
                    parts = line.split()
                    name = parts[0][1:]  # remove '>'
                    s_type = parts[1] if len(parts) > 1 else "NA"
                    p = ""  # reset sequence
                else:
                    p += line

            # Handle last sequence (EOF)
            if p and name and s_type:
                embedding = embedding_func(p.upper())
                out_file.write(f'{name},{s_type},{p},"{embedding}"\n')


def parse_embeddings_and_type(in_file):
    return pd.read_csv(
        in_file,
        converters={"embedding": ast.literal_eval},
        usecols=["type", "embedding"],
    )


def plot_umap_structural(df):
    scope_classes = {
        "a": "All-α proteins",
        "b": "All-β proteins",
        "c": "α/β proteins",
        "d": "α+β proteins",
        "e": "Multidomain proteins",
        "f": "Membrane and cell surface proteins",
        "g": "Small proteins",
    }

    # Extract class and embeddings
    df["class"] = df["type"].str[0]
    X = np.array(df["embedding"].tolist(), dtype=np.float32)

    # Standardize embeddings
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        random_state=None,
        init="random",  # <-- use random instead of spectral
        n_jobs=-1,
    )
    embedding_2d = reducer.fit_transform(X_scaled)
    print(pd.DataFrame(embedding_2d).head())

    # Plot
    plt.figure(figsize=(9, 7))
    unique_classes = sorted(df["class"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        idx = df["class"] == cls
        label = scope_classes.get(cls, f"Class {cls}")
        plt.scatter(
            embedding_2d[idx, 0],
            embedding_2d[idx, 1],
            color=colors[i],
            label=label,
            alpha=0.8,
            s=40,
            edgecolor="none",
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP of Protein Embeddings Colored by SCOPe Class")
    plt.legend(title="SCOPe Class", fontsize=9)
    plt.tight_layout()
    plt.show()
