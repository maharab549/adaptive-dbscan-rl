import os, sys
import numpy as np
from sklearn.datasets import make_blobs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adaptive_dbscan_rl.envs.dbscan_env import DBSCANParamEnv

def test_env_step():
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=0)
    env = DBSCANParamEnv(X)
    obs, _ = env.reset()
    d = X.shape[1]
    expected_len = 2 * min(3, d) + 2 * min(2, d)
    assert obs.shape[0] == expected_len
    state, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert terminated is True
    assert "eps" in info and "min_samples" in info
