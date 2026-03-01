import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    result = {"labels": labels}
    unique = np.unique(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(np.sum(unique != -1))
    result["n_noise"] = n_noise
    result["n_clusters"] = n_clusters
    if n_clusters > 1:
        result["silhouette"] = float(silhouette_score(X, labels))
        result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    else:
        result["silhouette"] = -1.0
        result["davies_bouldin"] = np.inf
        result["calinski_harabasz"] = 0.0
    return result
