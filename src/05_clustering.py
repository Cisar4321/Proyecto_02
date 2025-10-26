import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path

def run_kmeans(X, n_clusters=8):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X, n_clusters=8):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels, model

def run_gmm(X, n_clusters=8):
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def evaluate_clustering(X, labels):
    if len(set(labels)) <= 1:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski": np.nan}
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return {"silhouette": sil, "davies_bouldin": db, "calinski": ch}

def run_all_clusterings(X, n_clusters=8, eps=0.5, min_samples=5):
    results = {}
    algos = {
        "KMeans": lambda: run_kmeans(X, n_clusters),
        "DBSCAN": lambda: run_dbscan(X, eps, min_samples),
        "Agglomerative": lambda: run_agglomerative(X, n_clusters),
        "GMM": lambda: run_gmm(X, n_clusters),
    }

    for name, func in algos.items():
        print(f"\n--- {name} ---")
        labels, model = func()
        metrics = evaluate_clustering(X, labels)
        results[name] = metrics
        print(metrics)

    return results

if __name__ == "__main__":
    PATH_PCA = Path("data/X_pca.npy")
    PATH_SVD = Path("data/X_svd.npy")

    X_pca = np.load(PATH_PCA)
    X_svd = np.load(PATH_SVD)

    print("\n=== Resultados con PCA ===")
    res_pca = run_all_clusterings(X_pca, n_clusters=8, eps=0.8, min_samples=5)

    print("\n=== Resultados con SVD ===")
    res_svd = run_all_clusterings(X_svd, n_clusters=8, eps=0.8, min_samples=5)

    print("\nResumen PCA:")
    for k, v in res_pca.items():
        print(f"{k}: {v}")

    print("\nResumen SVD:")
    for k, v in res_svd.items():
        print(f"{k}: {v}")
