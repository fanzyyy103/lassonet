#!/usr/bin/env python

import numpy as np

from lassonet import LassoNetRegressor


def simulate_group_data(rng, *, n_per_group=200, p=12):
    X_all = []
    y_all = []
    groups = []
    for group in range(3):
        X = rng.normal(size=(n_per_group, p))
        common_signal = 2.0 * X[:, 0] - 1.6 * X[:, 1] + 1.2 * X[:, 2]
        group_signal = (
            (group == 0) * 2.5 * X[:, 3]
            + (group == 1) * 2.5 * X[:, 4]
            + (group == 2) * 2.5 * X[:, 5]
        )
        y = common_signal + group_signal + 0.6 * rng.normal(size=n_per_group)
        X_all.append(X)
        y_all.append(y)
        groups.append(np.full(n_per_group, group))
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(groups)


def choose_best_sparse_state(model, path, tolerance=0.10):
    """
    Choose a sparse model whose validation objective is close to the best one.
    tolerance=0.10 means allow validation objective to be within 10% of best.
    """
    valid_states = [state for state in path if state.state_dict is not None]
    if len(valid_states) == 0:
        valid_states = path

    best_val = min(state.val_objective for state in valid_states)
    candidates = [
        state
        for state in valid_states
        if state.val_objective <= best_val * (1 + tolerance)
    ]

    def n_selected_features(state):
        selected = np.asarray(state.selected)
        return int(selected.sum())

    best_sparse = min(candidates, key=n_selected_features)
    if best_sparse.state_dict is not None:
        model.load(best_sparse.state_dict)
    return best_sparse


def choose_state_with_target_sparsity(model, path, target_n_features=3):
    """
    Toy-example helper: choose the state whose number of selected features is
    closest to a desired sparsity target, then break ties with val_objective.
    """
    valid_states = [state for state in path if state.state_dict is not None]
    if len(valid_states) == 0:
        valid_states = path

    def score(state):
        n_selected = int(np.asarray(state.selected).sum())
        return (abs(n_selected - target_n_features), state.val_objective)

    best = min(valid_states, key=score)
    if best.state_dict is not None:
        model.load(best.state_dict)
    return best


def support_from_history_item(item):
    return np.flatnonzero(item.selected.cpu().numpy())


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean((y_true - y_pred) ** 2)


def summarize_path(name, path):
    print(f"\n{name} path summary:")
    for i, state in enumerate(path):
        selected = np.asarray(state.selected)
        print(
            i,
            "lambda =",
            state.lambda_,
            "val_objective =",
            state.val_objective,
            "n_selected =",
            int(selected.sum()),
            "selected =",
            [int(x) for x in np.where(selected)[0].tolist()],
        )


def main():
    rng = np.random.default_rng(0)
    alpha = 0.25
    true_common_features = [0, 1, 2]
    true_individual_features = {0: [3], 1: [4], 2: [5]}

    X_train, y_train, groups_train = simulate_group_data(rng, n_per_group=200, p=12)
    X_test, y_test, groups_test = simulate_group_data(rng, n_per_group=200, p=12)

    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True)
    feature_std[feature_std == 0] = 1.0
    X_train = (X_train - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    common_kwargs = dict(
        hidden_dims=(32,),
        lambda_seq=np.logspace(-3, 5, 80).tolist(),
        n_iters=(400, 150),
        patience=(50, 25),
        path_multiplier=1.05,
        val_size=0.2,
        random_state=0,
        torch_seed=0,
        verbose=0,
    )

    # Stage 1: fit a common LassoNet model on all groups.
    common_model = LassoNetRegressor(**common_kwargs)
    common_path = common_model.path(
        X_train,
        y_train,
        return_state_dicts=True,
    )
    summarize_path("Common", common_path)
    best_common = choose_state_with_target_sparsity(
        common_model,
        common_path,
        target_n_features=3,
    )
    selected_common_features = support_from_history_item(best_common)

    stage1_test_pred = common_model.predict(X_test).reshape(-1)
    stage1_test_mse = mean_squared_error(y_test, stage1_test_pred)

    final_test_pred = np.zeros_like(y_test, dtype=float)
    selected_individual_features = {}
    final_selected_features = {}

    stage2_kwargs = dict(
        hidden_dims=(32,),
        lambda_seq=np.logspace(-3, 3, 50).tolist(),
        n_iters=(400, 150),
        patience=(50, 25),
        path_multiplier=1.05,
        val_size=0.2,
        random_state=0,
        torch_seed=0,
        verbose=0,
    )

    for group in sorted(np.unique(groups_train)):
        train_mask = groups_train == group
        test_mask = groups_test == group

        f0_train = common_model.predict(X_train[train_mask]).reshape(-1)
        f0_test = common_model.predict(X_test[test_mask]).reshape(-1)
        offset_train = (1 - alpha) * f0_train
        offset_test = (1 - alpha) * f0_test

        penalty_factor = np.full(X_train.shape[1], 1.0 / alpha, dtype=np.float32)
        penalty_factor[selected_common_features] = 1.0

        # Stage 2: weighted LassoNet / pretrained LassoNet.
        # The only mathematical change is the per-feature threshold
        # lambda_j = lambda * penalty_factor[j] inside the hierarchical prox step.
        group_model = LassoNetRegressor(
            penalty_factor=penalty_factor,
            **stage2_kwargs,
        )
        group_path = group_model.path(
            X_train[train_mask],
            y_train[train_mask],
            offset=offset_train,
            return_state_dicts=True,
        )
        best_group = choose_best_sparse_state(
            group_model,
            group_path,
            tolerance=0.10,
        )

        weighted_support = support_from_history_item(best_group)
        selected_individual_features[group] = sorted(
            int(x) for x in (set(weighted_support) - set(selected_common_features))
        )
        final_selected_features[group] = sorted(
            int(x) for x in (set(weighted_support) | set(selected_common_features))
        )

        final_test_pred[test_mask] = offset_test + group_model.predict(X_test[test_mask]).reshape(-1)

    two_stage_test_mse = mean_squared_error(y_test, final_test_pred)

    # Compare weighted vs unweighted Stage 2 support on one group to show that
    # penalty_factor changes the selected features.
    compare_group = 0
    compare_mask = groups_train == compare_group
    compare_offset = (1 - alpha) * common_model.predict(X_train[compare_mask]).reshape(-1)

    weighted_compare_model = LassoNetRegressor(
        penalty_factor=np.where(
            np.isin(np.arange(X_train.shape[1]), selected_common_features),
            1.0,
            1.0 / alpha,
        ),
        **stage2_kwargs,
    )
    weighted_compare_path = weighted_compare_model.path(
        X_train[compare_mask],
        y_train[compare_mask],
        offset=compare_offset,
        return_state_dicts=True,
    )
    weighted_compare_best = choose_best_sparse_state(
        weighted_compare_model,
        weighted_compare_path,
        tolerance=0.10,
    )

    unweighted_compare_model = LassoNetRegressor(
        penalty_factor=np.ones(X_train.shape[1], dtype=np.float32),
        **stage2_kwargs,
    )
    unweighted_compare_path = unweighted_compare_model.path(
        X_train[compare_mask],
        y_train[compare_mask],
        offset=compare_offset,
        return_state_dicts=True,
    )
    unweighted_compare_best = choose_best_sparse_state(
        unweighted_compare_model,
        unweighted_compare_path,
        tolerance=0.10,
    )

    print("true common features:", true_common_features)
    print("selected common features:", selected_common_features.tolist())
    for group in sorted(true_individual_features):
        print(f"group {group} true individual features:", true_individual_features[group])
        print(
            f"group {group} selected individual features:",
            selected_individual_features[group],
        )
        print(
            f"group {group} final selected features:",
            final_selected_features[group],
        )

    print("Stage 1 test MSE:", float(stage1_test_mse))
    print("Two-stage test MSE:", float(two_stage_test_mse))
    print(
        "Group 0 selected features with pretrained penalty_factor:",
        support_from_history_item(weighted_compare_best).tolist(),
    )
    print(
        "Group 0 selected features with all-ones penalty_factor:",
        support_from_history_item(unweighted_compare_best).tolist(),
    )


if __name__ == "__main__":
    main()
