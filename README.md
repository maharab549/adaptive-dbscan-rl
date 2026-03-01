# Adaptive DBSCAN with Reinforcement Learning

Adaptive DBSCAN RL learns DBSCAN hyperparameters (`eps`, `min_samples`) from data characteristics using a lightweight RL policy. The project includes training, benchmarking, plotting utilities, and smoke tests.

## Features

- Learns DBSCAN parameters with a policy-gradient style loop
- Benchmarks against DBSCAN default, KMeans, OPTICS, and Agglomerative clustering
- Supports synthetic datasets (`blobs`, `moons`) out of the box
- Produces metrics and optional cluster visualizations
- Includes smoke tests for environment and training flow

## Project Structure

```
adaptive_dbscan_rl/
  agents/           # Policy and baseline models
  clustering/       # DBSCAN wrapper + metrics
  envs/             # RL environment for parameter selection
  training/         # Training loop
  utils/            # Reproducibility helpers
  visualization/    # Plotting helpers
scripts/
  run_train.py      # Train and save adaptive DBSCAN result
  run_benchmark.py  # Compare adaptive DBSCAN with baselines
tests/
  test_env_basic.py
  test_training_smoke.py
```

## Quickstart

### 1) Create environment and install dependencies

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Run training

```bash
python scripts/run_train.py --dataset moons --episodes 100 --samples 1000 --seed 42
```

Output is saved under `results/train_<dataset>.json`.

### 3) Run benchmark

```bash
python scripts/run_benchmark.py --dataset moons --episodes 100 --samples 1000 --plot
```

Output is saved under `results/benchmark_<dataset>.json` and plots (when `--plot` is provided).

### 4) Run tests

```bash
pytest -q
```

## Reproducibility

- Use `--seed` on scripts to control stochasticity.
- Global seeding is applied through `adaptive_dbscan_rl.utils.seeds.set_global_seed`.

## Notes

- The benchmark infers KMeans/Agglomerative `n_clusters` from adaptive DBSCAN cluster count.
- For very noisy or degenerate data, metrics may be limited (e.g., silhouette set to fallback value when only one cluster exists).