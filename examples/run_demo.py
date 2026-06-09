import numpy as np

from twostage_lassonet import TwoStagePretrainedLassoNetRegressor


def main():
    rng = np.random.default_rng(1234)
    n_per_group = 100
    p = 20

    X0 = rng.normal(size=(n_per_group, p)).astype(np.float32)
    X1 = rng.normal(size=(n_per_group, p)).astype(np.float32)
    X = np.vstack([X0, X1]).astype(np.float32)
    groups = np.array([0] * n_per_group + [1] * n_per_group)

    noise = 0.1 * rng.normal(size=2 * n_per_group)
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + (groups == 0) * 2.5 * X[:, 2]
        + (groups == 1) * 2.5 * X[:, 3]
        + noise
    ).astype(np.float32)

    model = TwoStagePretrainedLassoNetRegressor(
        alpha=0.5,
        stage1_lambda=1e-2,
        stage2_lambda=1e-2,
        common_model_kwargs={
            "hidden_dims": (16,),
            "dense_epochs": 100,
            "sparse_epochs": 60,
            "dense_patience": 20,
            "sparse_patience": 12,
            "random_state": 1234,
            "torch_seed": 1234,
            "verbose": 0,
        },
        group_model_kwargs={
            "hidden_dims": (16,),
            "dense_epochs": 100,
            "sparse_epochs": 60,
            "dense_patience": 20,
            "sparse_patience": 12,
            "random_state": 1234,
            "torch_seed": 1234,
            "verbose": 0,
        },
    )

    model.fit(X, y, groups)
    pred = model.predict(X, groups)
    mse = float(np.mean((pred - y) ** 2))

    print("Train MSE:", round(mse, 6))
    print("Common support:", model.get_common_support().tolist())
    print("Group 0 support:", model.get_group_support(0).tolist())
    print("Group 0 individual support:", model.get_group_individual_support(0).tolist())
    print("Group 0 final support:", model.get_group_final_support(0).tolist())
    print("Group 1 support:", model.get_group_support(1).tolist())
    print("Group 1 individual support:", model.get_group_individual_support(1).tolist())
    print("Group 1 final support:", model.get_group_final_support(1).tolist())


if __name__ == "__main__":
    main()
