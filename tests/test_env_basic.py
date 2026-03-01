import numpy as np
from sklearn.datasets import make_blobs
from adaptive_dbscan_rl.envs.dbscan_env import DBSCANParamEnv

def test_env_step():
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=0)
    env = DBSCANParamEnv(X)
    obs, _ = env.reset()
    assert obs.shape[0] == 10
    state, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert terminated is True
    assert "eps" in info and "min_samples" in info
