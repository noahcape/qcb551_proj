import pytest
from main import Clustering, Similarity, create_similarity_matrix, cluster

example_df = [
    [0.12, -0.48, 0.30, 0.89, -0.22, 0.05, 0.61, -0.73],  # MTEITAAMVKELRESTGAGM
    [0.45, 0.11, -0.62, 0.33, 0.19, -0.27, -0.15, 0.84],  # GAVLILKKKGEANVKFS
    [-0.32, 0.76, 0.08, -0.40, 0.57, -0.12, 0.29, 0.61],  # MKKLLFTTAAFFLALAGASA
    [0.23, -0.05, 0.41, 0.67, -0.10, -0.91, 0.24, 0.12],  # VTILGEEALFSDQCME
    [-0.15, 0.38, -0.47, 0.52, -0.01, 0.23, 0.80, -0.36],  # AFKEHQLGNVTKPS
]


@pytest.mark.parametrize(
    "similarity, clustering",
    [
        (Similarity.Cosine, Clustering.Hierarchical),
        (Similarity.Cosine, Clustering.Kmeans),
        (Similarity.Cosine, Clustering.Spectral),
        (Similarity.Euclidean, Clustering.Hierarchical),
        (Similarity.Euclidean, Clustering.Kmeans),
        (Similarity.Euclidean, Clustering.Spectral),
        (Similarity.Geodesic, Clustering.Hierarchical),
        (Similarity.Geodesic, Clustering.Kmeans),
        (Similarity.Geodesic, Clustering.Spectral),
    ],
)
def test_all(similarity, clustering):
    print(similarity, clustering)
    sim_m = create_similarity_matrix(similarity, example_df)
    clusters = cluster(sim_m, clustering, 2)
    print(clusters)
