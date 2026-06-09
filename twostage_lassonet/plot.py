import os
from pathlib import Path

_mplconfig_dir = Path.cwd() / ".mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_path_summary(path_items, output_path, *, score_label="score"):
    if not path_items:
        raise ValueError("path_items must not be empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lambdas = np.array([item.lambda_ for item in path_items], dtype=float)
    scores = np.array([item.score for item in path_items], dtype=float)
    n_selected = np.array([item.n_selected_features for item in path_items], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(16, 16))

    axes[0].grid(True, alpha=0.7)
    axes[0].plot(n_selected, scores, ".-", linewidth=1.5, markersize=5)
    axes[0].set_xlabel("number of selected features")
    axes[0].set_ylabel(score_label)

    axes[1].grid(True, alpha=0.7)
    axes[1].plot(lambdas, scores, ".-", linewidth=1.5, markersize=4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel(score_label)

    axes[2].grid(True, alpha=0.7)
    axes[2].plot(lambdas, n_selected, ".-", linewidth=1.5, markersize=4)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("number of selected features")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
