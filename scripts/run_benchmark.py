import argparse
import json
import os
import sys
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adaptive_dbscan_rl.training.train import train_agents
from adaptive_dbscan_rl.visualization.plotting import plot_clusters
from adaptive_dbscan_rl.utils.seeds import set_global_seed

def load_data(name: str, n_samples: int, seed: int) -> np.ndarray:
    if name == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.2, random_state=seed)
        return X
    if name == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.08, random_state=seed)
        return X
    raise ValueError("unknown dataset")

def metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    unique = np.unique(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(np.sum(unique != -1))
    if n_clusters <= 1:
        return {"silhouette": -1.0, "davies_bouldin": float("inf"), "calinski_harabasz": 0.0, "n_noise": n_noise, "n_clusters": n_clusters}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "n_noise": n_noise,
        "n_clusters": n_clusters,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="blobs")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    set_global_seed(args.seed)
    X = load_data(args.dataset, args.samples, args.seed)
    best_params, _, _ = train_agents(X, episodes=args.episodes, seed=args.seed)
    labels_adapt = DBSCAN(eps=best_params[0], min_samples=best_params[1]).fit_predict(X)
    labels_dbscan = DBSCAN().fit_predict(X)
    labels_kmeans = KMeans(n_clusters=len(np.unique(labels_adapt)) - (1 if -1 in labels_adapt else 0), n_init="auto", random_state=args.seed).fit_predict(X)
    labels_optics = OPTICS().fit_predict(X)
    labels_agg = AgglomerativeClustering(n_clusters=len(np.unique(labels_adapt)) - (1 if -1 in labels_adapt else 0)).fit_predict(X)
    results = {
        "adaptive_dbscan": {"params": {"eps": float(best_params[0]), "min_samples": int(best_params[1])}, "metrics": metrics(X, labels_adapt)},
        "dbscan_default": {"metrics": metrics(X, labels_dbscan)},
        "kmeans": {"metrics": metrics(X, labels_kmeans)},
        "optics": {"metrics": metrics(X, labels_optics)},
        "agglomerative": {"metrics": metrics(X, labels_agg)},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"benchmark_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if args.plot:
        plot_clusters(X, labels_adapt, "Adaptive DBSCAN", os.path.join(args.out, f"adaptive_{args.dataset}.png"))
        plot_clusters(X, labels_dbscan, "DBSCAN Default", os.path.join(args.out, f"dbscan_{args.dataset}.png"))
        plot_clusters(X, labels_kmeans, "KMeans", os.path.join(args.out, f"kmeans_{args.dataset}.png"))
        plot_clusters(X, labels_optics, "OPTICS", os.path.join(args.out, f"optics_{args.dataset}.png"))
        plot_clusters(X, labels_agg, "Agglomerative", os.path.join(args.out, f"agg_{args.dataset}.png"))

if __name__ == "__main__":
    main()
