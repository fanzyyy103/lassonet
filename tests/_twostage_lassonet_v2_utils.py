import numpy as np


def make_small_group_data(seed=7, n_per_group=36, p=8):
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []
    group_parts = []
    for group, shift in [("G1", 0.7), ("G2", -0.6)]:
        X = rng.normal(size=(n_per_group, p)).astype(np.float32)
        shared = 1.2 * X[:, 0] - 0.8 * X[:, 1] + np.sin(X[:, 2])
        correction = shift * X[:, 3] + 0.3 * X[:, 4]
        noise = rng.normal(scale=0.15, size=n_per_group)
        y = shared + correction + noise
        X_parts.append(X)
        y_parts.append(y.astype(np.float32))
        group_parts.extend([group] * n_per_group)
    return np.vstack(X_parts), np.concatenate(y_parts), np.asarray(group_parts)


def tiny_estimator_kwargs(**overrides):
    params = {
        "hidden_dims_stage1": (4,),
        "hidden_dims_stage2": (4,),
        "alpha_values": [0.0, 0.5, 1.0],
        "lambda_stage1": [0.001, 0.01],
        "lambda_stage2": [0.001, 0.01],
        "validation_fraction": 0.25,
        "random_state": 123,
        "refit": False,
        "verbose": 0,
        "n_iters": (8, 3),
        "patience": (3, 1),
        "backtrack": True,
        "path_multiplier": 1.1,
    }
    params.update(overrides)
    return params
