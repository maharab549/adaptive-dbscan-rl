import os, sys
import numpy as np
from sklearn.datasets import make_moons
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adaptive_dbscan_rl.training.train import train_agents
from adaptive_dbscan_rl.clustering.dbscan_wrapper import run_dbscan

def test_training_smoke():
    X, _ = make_moons(n_samples=300, noise=0.08, random_state=7)
    best_params, _, _ = train_agents(X, episodes=10, seed=7)
    assert float(best_params[0]) > 0
    assert int(best_params[1]) >= 1
    res = run_dbscan(X, best_params[0], best_params[1])
    assert "labels" in res and "n_clusters" in res and "n_noise" in res
