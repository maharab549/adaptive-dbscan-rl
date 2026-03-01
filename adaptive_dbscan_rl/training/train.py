import numpy as np
import torch
import torch.optim as optim
from typing import Tuple
from adaptive_dbscan_rl.agents.policy import Actor, Baseline
from adaptive_dbscan_rl.envs.dbscan_env import DBSCANParamEnv

def train_agents(X: np.ndarray, episodes: int = 200, lr: float = 1e-3, seed: int = 42) -> Tuple[Tuple[float, int], Actor, Actor]:
    torch.manual_seed(seed)
    env = DBSCANParamEnv(X)
    obs, _ = env.reset()
    obs_dim = int(obs.shape[0])
    actor_eps = Actor(obs_dim, out_dim=1)
    actor_ms = Actor(obs_dim, out_dim=1)
    baseline = Baseline(obs_dim)
    opt = optim.Adam(list(actor_eps.parameters()) + list(actor_ms.parameters()) + list(baseline.parameters()), lr=lr)
    best_params = None
    best_reward = -1e9
    for _ in range(episodes):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        eps_a = actor_eps(obs_t)
        ms_a = actor_ms(obs_t)
        action = torch.cat([eps_a, ms_a], dim=1).detach().numpy()[0]
        _, reward, _, _, info = env.step(action)
        value = baseline(obs_t).squeeze(0)
        adv = torch.tensor([reward], dtype=torch.float32) - value
        loss = -adv * (eps_a.log() + (1.0 - eps_a).log() + ms_a.log() + (1.0 - ms_a).log()).mean() + (adv.pow(2)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if reward > best_reward:
            best_reward = reward
            best_params = (info["eps"], info["min_samples"])
    return best_params, actor_eps, actor_ms
