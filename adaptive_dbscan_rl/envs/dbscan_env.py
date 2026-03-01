import numpy as np
try:
    from gym import Env, spaces
except Exception:
    from gymnasium import Env, spaces
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN

class DBSCANParamEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, X: np.ndarray, eps_bounds: tuple[float, float] | None = None, min_samples_bounds: tuple[int, int] = (1, 50)):
        super().__init__()
        self.X = X
        self.n_samples = X.shape[0]
        self.eps_bounds = eps_bounds or self._auto_eps_bounds()
        self.min_samples_bounds = min_samples_bounds
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.state = None
        self.best_reward = -np.inf
        self.best_params = None

    def _auto_eps_bounds(self) -> tuple[float, float]:
        dists = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=-1)
        tri = dists[np.triu_indices(self.n_samples, k=1)]
        low = float(np.quantile(tri, 0.01))
        high = float(np.quantile(tri, 0.2))
        return max(1e-8, low), max(low + 1e-8, high)

    def _summarize(self) -> np.ndarray:
        x = self.X
        flat = np.concatenate([
            np.mean(x, axis=0)[:3],
            np.std(x, axis=0)[:3],
            np.quantile(x, 0.25, axis=0)[:2],
            np.quantile(x, 0.75, axis=0)[:2],
        ])
        return flat.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.state = self._summarize()
        return self.state, {}

    def step(self, action: np.ndarray):
        a = np.clip(action, 0.0, 1.0)
        eps = self.eps_bounds[0] + a[0] * (self.eps_bounds[1] - self.eps_bounds[0])
        ms = int(self.min_samples_bounds[0] + a[1] * (self.min_samples_bounds[1] - self.min_samples_bounds[0]))
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(self.X)
        reward = self._reward(labels)
        terminated = True
        truncated = False
        info = {"eps": eps, "min_samples": ms, "labels": labels}
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = (eps, ms)
        return self.state, float(reward), terminated, truncated, info

    def _reward(self, labels: np.ndarray) -> float:
        unique = np.unique(labels)
        n_noise = int(np.sum(labels == -1))
        n_clusters = int(np.sum(unique != -1))
        if n_clusters <= 1:
            return -1.0 - 0.5 * (n_noise / self.n_samples)
        sil = silhouette_score(self.X, labels)
        db = davies_bouldin_score(self.X, labels)
        ch = calinski_harabasz_score(self.X, labels)
        noise_penalty = n_noise / self.n_samples
        score = 0.6 * sil + 0.2 * (ch / (ch + 1.0)) - 0.2 * (db / (db + 1.0)) - 0.3 * noise_penalty
        return float(score)
