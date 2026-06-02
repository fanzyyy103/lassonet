#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

_mplconfig_dir = Path.cwd() / ".mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lassonet import LassoNetRegressor


def simulate_different_groups_data(rng, *, n_per_group=150, p=50):
    """
    Mimic the DifferentGroupsTrainAndTest.Rmd setup:
    - training groups: 0, 1, 2
    - test groups: 3, 4
    - first 3 features are shared support
    - each training group has its own group-specific block
    - test groups are mixtures of training groups through hidden labels
    """
    n_groups = 5
    n = n_groups * n_per_group
    observed_groups = np.repeat(np.arange(n_groups), n_per_group)

    beta_group0 = np.zeros(p)
    beta_group1 = np.zeros(p)
    beta_group2 = np.zeros(p)
    beta_group0[:3] = [-1.0, 1.0, 1.0]
    beta_group1[:3] = [-1.0, 1.0, 1.0]
    beta_group2[:3] = [-1.0, 1.0, 1.0]
    beta_group0[3:6] = 0.5
    beta_group1[6:9] = 0.5
    beta_group2[9:12] = 0.5

    hidden_groups = observed_groups.copy()
    hidden_groups[observed_groups == 3] = rng.choice(
        [0, 1],
        size=np.sum(observed_groups == 3),
        replace=True,
    )
    hidden_groups[observed_groups == 4] = rng.choice(
        [0, 2],
        size=np.sum(observed_groups == 4),
        replace=True,
    )

    X = rng.normal(size=(n, p))
    X[hidden_groups == 0, :3] += 1.0
    X[hidden_groups == 1, :3] += 2.0
    X[hidden_groups == 2, :3] += 3.0

    y_signal = np.zeros(n, dtype=np.float64)
    y_signal[hidden_groups == 0] = np.sum(
        X[hidden_groups == 0, :6] * beta_group0[:6],
        axis=1,
    )
    y_signal[hidden_groups == 1] = np.sum(
        X[hidden_groups == 1, :9] * beta_group1[:9],
        axis=1,
    )
    y_signal[hidden_groups == 2] = np.sum(
        X[hidden_groups == 2, :12] * beta_group2[:12],
        axis=1,
    )
    y = y_signal + 2.5 * rng.normal(size=n)

    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.float32),
        "observed_groups": observed_groups,
        "hidden_groups": hidden_groups,
        "beta_group0": beta_group0,
        "beta_group1": beta_group1,
        "beta_group2": beta_group2,
        "true_common_features": [0, 1, 2],
        "true_individual_features": {
            0: [3, 4, 5],
            1: [6, 7, 8],
            2: [9, 10, 11],
        },
    }


def split_train_val_test(data, rng):
    observed_groups = data["observed_groups"]
    train_pool_mask = observed_groups < 3
    pool_indices = np.where(train_pool_mask)[0]
    val_size = len(pool_indices) // 3
    val_indices = rng.choice(pool_indices, size=val_size, replace=False)
    val_mask = np.zeros(len(observed_groups), dtype=bool)
    val_mask[val_indices] = True

    train_mask = train_pool_mask & ~val_mask
    test_mask = observed_groups >= 3

    return {
        "X_train": data["X"][train_mask],
        "y_train": data["y"][train_mask],
        "groups_train": observed_groups[train_mask],
        "X_val": data["X"][val_mask],
        "y_val": data["y"][val_mask],
        "groups_val": observed_groups[val_mask],
        "X_test": data["X"][test_mask],
        "y_test": data["y"][test_mask],
        "groups_test": observed_groups[test_mask],
        "hidden_groups_test": data["hidden_groups"][test_mask],
    }


def standardize_with_train(X_train, X_val, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
    )


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def support_from_history_item(item):
    return [int(x) for x in np.flatnonzero(item.selected.cpu().numpy()).tolist()]


def choose_best_by_val_loss(model, path):
    valid_states = [state for state in path if state.state_dict is not None]
    best = min(valid_states, key=lambda state: state.val_loss)
    model.load(best.state_dict)
    return best


def eval_path_mse(model, path, X_eval, y_eval):
    lambdas = []
    mses = []
    n_selected = []
    current_state = model.model.cpu_state_dict() if model.model is not None else None
    for state in path:
        if state.state_dict is None:
            continue
        model.load(state.state_dict)
        predictions = model.predict(X_eval).reshape(-1)
        lambdas.append(float(state.lambda_))
        mses.append(mean_squared_error(y_eval, predictions))
        n_selected.append(int(np.asarray(state.selected).sum()))
    if current_state is not None:
        model.load(current_state)
    return np.array(lambdas), np.array(mses), np.array(n_selected)


def save_path_plot(
    model,
    path,
    X_eval,
    y_eval,
    output_path,
    *,
    title,
    selected_lambda=None,
):
    lambdas, mses, n_selected = eval_path_mse(model, path, X_eval, y_eval)
    selected_idx = None
    if selected_lambda is not None and len(lambdas) > 0:
        selected_idx = int(np.argmin(np.abs(lambdas - selected_lambda)))

    fig = plt.figure(figsize=(14, 14))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, mses, ".-")
    if selected_idx is not None:
        plt.scatter(
            [n_selected[selected_idx]],
            [mses[selected_idx]],
            color="tab:red",
            s=80,
            zorder=3,
            label="selected lambda",
        )
        plt.legend()
    plt.xlabel("number of selected features")
    plt.ylabel("validation MSE")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambdas, mses, ".-")
    if selected_idx is not None:
        plt.axvline(selected_lambda, color="tab:red", linestyle="--", linewidth=1.5)
        plt.scatter(
            [lambdas[selected_idx]],
            [mses[selected_idx]],
            color="tab:red",
            s=80,
            zorder=3,
            label="selected lambda",
        )
        plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("validation MSE")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambdas, n_selected, ".-")
    if selected_idx is not None:
        plt.axvline(selected_lambda, color="tab:red", linestyle="--", linewidth=1.5)
        plt.scatter(
            [lambdas[selected_idx]],
            [n_selected[selected_idx]],
            color="tab:red",
            s=80,
            zorder=3,
            label="selected lambda",
        )
        plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(1234)
    alpha = 0.5
    output_dir = Path("outputs") / "pretrained_lassonet_different_groups"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = simulate_different_groups_data(rng, n_per_group=150, p=50)
    split = split_train_val_test(data, rng)

    X_train, X_val, X_test = standardize_with_train(
        split["X_train"],
        split["X_val"],
        split["X_test"],
    )
    y_train = split["y_train"]
    y_val = split["y_val"]
    y_test = split["y_test"]
    groups_train = split["groups_train"]
    groups_val = split["groups_val"]
    hidden_groups_test = split["hidden_groups_test"]

    common_kwargs = dict(
        hidden_dims=(32,),
        lambda_seq=np.logspace(-3, 4, 40).tolist(),
        n_iters=(300, 120),
        patience=(40, 20),
        val_size=0.0,
        random_state=1234,
        torch_seed=1234,
        verbose=0,
    )

    stage1_model = LassoNetRegressor(**common_kwargs)
    stage1_path = stage1_model.path(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        return_state_dicts=True,
    )
    best_stage1 = choose_best_by_val_loss(stage1_model, stage1_path)
    selected_common_features = support_from_history_item(best_stage1)
    stage1_lambda = float(best_stage1.lambda_)

    save_path_plot(
        stage1_model,
        stage1_path,
        X_val,
        y_val,
        output_dir / "stage1_lambda_path.png",
        title="Stage 1 Common LassoNet Path",
        selected_lambda=stage1_lambda,
    )

    # Similarity model: mimic the Rmd by predicting which training group
    # an observation resembles.
    similarity_model = LogisticRegression(max_iter=2000, random_state=1234)
    similarity_model.fit(X_train, groups_train)
    class_probs_test = similarity_model.predict_proba(X_test)

    stage1_test_pred = stage1_model.predict(X_test).reshape(-1)
    stage1_test_mse = mean_squared_error(y_test, stage1_test_pred)

    stage2_group_models = {}
    stage2_lambdas = {}
    stage2_selected_features = {}
    pretrained_group_predictions = []

    offset_test = (1 - alpha) * stage1_model.predict(X_test).reshape(-1)

    for group in sorted(np.unique(groups_train)):
        train_mask = groups_train == group
        val_mask = groups_val == group

        offset_train = (1 - alpha) * stage1_model.predict(X_train[train_mask]).reshape(-1)
        offset_val = (1 - alpha) * stage1_model.predict(X_val[val_mask]).reshape(-1)

        penalty_factor = np.full(X_train.shape[1], 1.0 / alpha, dtype=np.float32)
        penalty_factor[selected_common_features] = 1.0

        stage2_model = LassoNetRegressor(
            penalty_factor=penalty_factor,
            hidden_dims=(32,),
            lambda_seq=np.logspace(-3, 4, 40).tolist(),
            n_iters=(300, 120),
            patience=(40, 20),
            val_size=0.0,
            random_state=1234,
            torch_seed=1234,
            verbose=0,
        )
        stage2_path = stage2_model.path(
            X_train[train_mask],
            y_train[train_mask],
            X_val=X_val[val_mask],
            y_val=y_val[val_mask],
            offset=offset_train,
            offset_val=offset_val,
            return_state_dicts=True,
        )
        best_stage2 = choose_best_by_val_loss(stage2_model, stage2_path)

        stage2_group_models[group] = stage2_model
        stage2_lambdas[group] = float(best_stage2.lambda_)
        stage2_selected_features[group] = support_from_history_item(best_stage2)

        save_path_plot(
            stage2_model,
            stage2_path,
            X_val[val_mask],
            y_val[val_mask] - offset_val,
            output_dir / f"stage2_group_{group}_lambda_path.png",
            title=f"Stage 2 Group {group} LassoNet Path",
            selected_lambda=stage2_lambdas[group],
        )

        residual_test_pred = stage2_model.predict(X_test).reshape(-1)
        pretrained_group_predictions.append(offset_test + residual_test_pred)

    pretrained_group_predictions = np.column_stack(pretrained_group_predictions)
    two_stage_test_pred = np.sum(pretrained_group_predictions * class_probs_test, axis=1)
    two_stage_test_mse = mean_squared_error(y_test, two_stage_test_pred)

    print("alpha:", alpha)
    print("true common features:", data["true_common_features"])
    print("selected common features:", selected_common_features)
    print("stage 1 selected lambda:", stage1_lambda)
    for group in sorted(stage2_group_models):
        print(f"group {group} true individual features:", data["true_individual_features"][group])
        print(f"group {group} selected stage 2 features:", stage2_selected_features[group])
        print(f"group {group} selected lambda:", stage2_lambdas[group])

    print("Stage 1 test MSE:", stage1_test_mse)
    print("Two-stage weighted test MSE:", two_stage_test_mse)
    print(
        "saved plots:",
        [
            str(output_dir / "stage1_lambda_path.png"),
            *[
                str(output_dir / f"stage2_group_{group}_lambda_path.png")
                for group in sorted(stage2_group_models)
            ],
        ],
    )


if __name__ == "__main__":
    main()
