import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(X: np.ndarray, labels: np.ndarray, title: str, path: str | None = None) -> None:
    sns.set(style="white", context="talk")
    unique = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique))
    colors = {u: palette[i % len(palette)] for i, u in enumerate(unique) if u != -1}
    noise_color = (0.1, 0.1, 0.1)
    for u in unique:
        m = labels == u
        c = colors[u] if u != -1 else noise_color
        plt.scatter(X[m, 0], X[m, 1], s=10, color=c)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
