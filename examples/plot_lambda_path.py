from pathlib import Path

import numpy as np

from twostage_lassonet import (
    fit_two_stage_lambda_path,
    plot_path_summary,
)


def main():
    rng = np.random.default_rng(1234)
    n_per_group = 100
    p = 10

    X0 = rng.normal(size=(n_per_group, p)).astype(np.float32)
    X1 = rng.normal(size=(n_per_group, p)).astype(np.float32)
    X = np.vstack([X0, X1]).astype(np.float32)
    groups = np.array([0] * n_per_group + [1] * n_per_group)

    noise = 0.15 * rng.normal(size=2 * n_per_group)
    y = (
        2.0 * X[:, 0]
        - 1.4 * X[:, 1]
        + (groups == 0) * 2.0 * X[:, 2]
        + (groups == 1) * 2.0 * X[:, 3]
        + noise
    ).astype(np.float32)

    lambda_seq = np.logspace(np.log10(3.0), np.log10(1200.0), 140)

    path_items = fit_two_stage_lambda_path(
        X,
        y,
        groups,
        lambda_seq=lambda_seq,
        alpha=0.5,
        stage1_lambda=1e-2,
        common_model_kwargs={
            "hidden_dims": (16,),
            "dense_epochs": 80,
            "sparse_epochs": 40,
            "dense_patience": 15,
            "sparse_patience": 10,
            "random_state": 1234,
            "torch_seed": 1234,
            "verbose": 0,
        },
        group_model_kwargs={
            "hidden_dims": (16,),
            "dense_epochs": 80,
            "sparse_epochs": 40,
            "dense_patience": 15,
            "sparse_patience": 10,
            "random_state": 1234,
            "torch_seed": 1234,
            "verbose": 0,
        },
    )

    output_path = Path("outputs") / "lambda_path_summary.png"
    plot_path_summary(path_items, output_path, score_label="score")
    print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
