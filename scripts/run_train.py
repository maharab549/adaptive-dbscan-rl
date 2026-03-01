import argparse
import json
import os
import sys
import numpy as np
from sklearn.datasets import make_blobs, make_moons
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adaptive_dbscan_rl.training.train import train_agents
from adaptive_dbscan_rl.clustering.dbscan_wrapper import run_dbscan
from adaptive_dbscan_rl.utils.seeds import set_global_seed

def load_data(name: str, n_samples: int, seed: int) -> np.ndarray:
    if name == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.2, random_state=seed)
        return X
    if name == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.08, random_state=seed)
        return X
    raise ValueError("unknown dataset")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="blobs")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    set_global_seed(args.seed)
    X = load_data(args.dataset, args.samples, args.seed)
    best_params, _, _ = train_agents(X, episodes=args.episodes, seed=args.seed)
    result = run_dbscan(X, best_params[0], best_params[1])
    if "labels" in result:
        result["labels"] = result["labels"].tolist()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"train_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump({"best_eps": float(best_params[0]), "best_min_samples": int(best_params[1]), "metrics": result}, f, indent=2)

if __name__ == "__main__":
    main()
