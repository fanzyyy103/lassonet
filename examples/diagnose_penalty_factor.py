#!/usr/bin/env python

import numpy as np
import torch

from lassonet import LassoNetRegressor
from lassonet.model import _apply_penalty_factor


def simulate_data(rng, *, n=240, p=6):
    X = rng.normal(size=(n, p))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.25 * rng.normal(size=n)
    return X.astype(np.float32), y.astype(np.float32)


def summarize_support_path(name, path):
    print(f"\n{name} support path:")
    for i, state in enumerate(path):
        selected = np.asarray(state.selected.cpu().numpy(), dtype=bool)
        print(
            i,
            "lambda =",
            state.lambda_,
            "n_selected =",
            int(selected.sum()),
            "selected =",
            [int(x) for x in np.where(selected)[0].tolist()],
        )


def choose_sparsest_nonempty_state(model, path):
    valid_states = [state for state in path if state.state_dict is not None]
    nonempty_states = [
        state for state in valid_states if int(np.asarray(state.selected).sum()) > 0
    ]
    candidates = nonempty_states if nonempty_states else valid_states
    best = min(candidates, key=lambda state: (int(np.asarray(state.selected).sum()), state.val_objective))
    model.load(best.state_dict)
    return best


def main():
    rng = np.random.default_rng(0)
    X, y = simulate_data(rng)

    # Check 1: did penalty_factor actually get passed into the estimator?
    penalty_factor = np.ones(X.shape[1], dtype=np.float32)
    penalty_factor[0] = 1000.0

    individual_model = LassoNetRegressor(
        penalty_factor=penalty_factor,
        hidden_dims=(16,),
        lambda_seq=np.logspace(-3, 4, 30).tolist(),
        n_iters=(200, 80),
        patience=(20, 10),
        val_size=0.2,
        random_state=0,
        torch_seed=0,
        verbose=0,
    )

    print("CHECK 1")
    print("penalty_factor used (constructor):", individual_model.penalty_factor)

    path = individual_model.path(X, y, return_state_dicts=True)
    print("penalty_factor_ used internally (torch tensor):", individual_model.penalty_factor_)

    # Check 2: did penalty_factor actually enter the weighted penalty and weighted prox threshold?
    choose_sparsest_nonempty_state(individual_model, path)
    column_norms = torch.norm(individual_model.model.skip.weight.data, p=2, dim=0)
    weighted_regularization = torch.sum(column_norms * individual_model.penalty_factor_).item()
    unweighted_regularization = torch.sum(column_norms).item()
    lambda_probe = torch.tensor(1.0, device=individual_model.device)
    weighted_lambda = _apply_penalty_factor(
        lambda_probe,
        individual_model.penalty_factor_,
        ref=individual_model.model.skip.weight.data,
    ).detach().cpu().numpy()

    print("\nCHECK 2")
    print("skip column norms:", column_norms.detach().cpu().numpy())
    print(
        "weighted penalty = sum_j penalty_factor[j] * ||skip_j||_2 =",
        weighted_regularization,
    )
    print(
        "unweighted penalty = sum_j ||skip_j||_2 =",
        unweighted_regularization,
    )
    print("weighted lambda thresholds for lambda = 1.0:", weighted_lambda)
    print(
        "Note: training loss still uses the original LassoNet pattern "
        "(fit loss + Hier-Prox step). The weighted penalty enters through "
        "l1_regularization_skip(...) and through weighted Hier-Prox thresholds."
    )

    # Check 3: extreme test. Give feature 0 a huge penalty and compare support.
    common_kwargs = dict(
        hidden_dims=(16,),
        lambda_seq=np.logspace(-3, 4, 30).tolist(),
        n_iters=(200, 80),
        patience=(20, 10),
        val_size=0.2,
        random_state=0,
        torch_seed=0,
        verbose=0,
    )

    unweighted_model = LassoNetRegressor(**common_kwargs)
    unweighted_path = unweighted_model.path(X, y, return_state_dicts=True)
    unweighted_best = choose_sparsest_nonempty_state(unweighted_model, unweighted_path)

    extreme_penalty = np.ones(X.shape[1], dtype=np.float32)
    extreme_penalty[0] = 1_000_000.0
    weighted_model = LassoNetRegressor(
        penalty_factor=extreme_penalty,
        **common_kwargs,
    )
    weighted_path = weighted_model.path(X, y, return_state_dicts=True)
    weighted_best = choose_sparsest_nonempty_state(weighted_model, weighted_path)

    print("\nCHECK 3")
    summarize_support_path("Unweighted", unweighted_path)
    summarize_support_path("Extreme weighted (feature 0 heavily penalized)", weighted_path)
    print(
        "best unweighted selected features:",
        [int(x) for x in np.where(np.asarray(unweighted_best.selected))[0].tolist()],
    )
    print(
        "best weighted selected features:",
        [int(x) for x in np.where(np.asarray(weighted_best.selected))[0].tolist()],
    )


if __name__ == "__main__":
    main()
