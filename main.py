from enum import Enum
import scipy as sc
import sklearn as sk


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
