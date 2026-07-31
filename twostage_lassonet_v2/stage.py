from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, List, Sequence

import numpy as np

from lassonet import LassoNetRegressor

from .results import Stage1Result, Stage2GroupResult
from .selection import decompose_support
from .utils import mse, prediction_vector, support_from_model, theta_from_model


class FeatureSubsetModelAdapter:
    """Expose a subset-trained LassoNet model as a full-width correction model."""

    def __init__(self, estimator, feature_indices: Sequence[int], n_features: int):
        self.estimator = estimator
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.n_features = int(n_features)
        self.model = getattr(estimator, "model", None)

    def predict(self, X):
        X_array = np.asarray(X, dtype=np.float32)
        return self.estimator.predict(X_array[:, self.feature_indices])

    def load(self, state_dict):
        self.estimator.load(state_dict)
        self.model = getattr(self.estimator, "model", None)
        return self


def _new_lassonet_model(
    model_kwargs,
    *,
    hidden_dims,
    M,
    seed,
    verbose,
    estimator_cls=LassoNetRegressor,
):
    kwargs = deepcopy(model_kwargs or {})
    kwargs.setdefault("hidden_dims", tuple(hidden_dims))
    kwargs.setdefault("M", M)
    kwargs.setdefault("verbose", verbose)
    kwargs.setdefault("random_state", seed)
    kwargs.setdefault("torch_seed", seed)
    kwargs.setdefault("gamma", 0.0)
    kwargs.setdefault("gamma_skip", 0.0)
    return estimator_cls(**kwargs)


def _candidate_items(path: List[Any]):
    if len(path) > 1 and float(path[0].lambda_) == 0.0:
        return path[1:]
    return path


def _select_history_by_val_loss(path: List[Any]):
    candidates = _candidate_items(path)
    return min(candidates, key=lambda item: float(item.val_loss))


def fit_stage1(
    X_train,
    y_train,
    *,
    X_val,
    y_val,
    lambda_seq,
    hidden_dims,
    M,
    support_tol,
    model_kwargs,
    seed,
    verbose,
    estimator_cls=LassoNetRegressor,
    prefer_input_mask=True,
):
    model = _new_lassonet_model(
        model_kwargs,
        hidden_dims=hidden_dims,
        M=M,
        seed=seed,
        verbose=verbose,
        estimator_cls=estimator_cls,
    )
    path = model.path(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        lambda_seq=lambda_seq,
        return_state_dicts=True,
    )
    selected = _select_history_by_val_loss(path)
    model.load(selected.state_dict)
    theta = theta_from_model(model, X_train.shape[1])
    support = support_from_model(
        model,
        X_train.shape[1],
        support_tol=support_tol,
        prefer_input_mask=prefer_input_mask,
    )
    return Stage1Result(
        model=model,
        selected_lambda=float(selected.lambda_),
        theta=theta,
        support=support,
        path=path,
        validation_scores=np.asarray([float(item.val_loss) for item in path], dtype=np.float64),
    )


def fit_stage2_group(
    X_train,
    y_train,
    *,
    X_val,
    y_val,
    offset_train,
    offset_val,
    group,
    alpha,
    common_support,
    feature_indices,
    penalty_factor,
    lambda_seq,
    hidden_dims,
    M,
    support_tol,
    model_kwargs,
    seed,
    verbose,
    estimator_cls=LassoNetRegressor,
    validation_metric: Callable[[np.ndarray, np.ndarray], float] = mse,
    prefer_input_mask=True,
):
    n_features = common_support.shape[0]
    feature_indices = np.asarray(feature_indices, dtype=int)
    X_train_subset = X_train[:, feature_indices]
    X_val_subset = X_val[:, feature_indices]
    subset_penalty = np.asarray(penalty_factor, dtype=np.float32)[feature_indices]

    if not np.all(np.isfinite(subset_penalty)):
        raise ValueError("Stage 2 penalty factors must be finite.")
    if np.any(subset_penalty < 0.0):
        raise ValueError("Stage 2 penalty factors must be nonnegative.")

    estimator = _new_lassonet_model(
        model_kwargs,
        hidden_dims=hidden_dims,
        M=M,
        seed=seed,
        verbose=verbose,
        estimator_cls=estimator_cls,
    )
    estimator.penalty_factor = subset_penalty
    path = estimator.path(
        X_train_subset,
        y_train,
        X_val=X_val_subset,
        y_val=y_val,
        offset=offset_train,
        offset_val=offset_val,
        lambda_seq=lambda_seq,
        return_state_dicts=True,
    )

    best_item = None
    best_validation_loss = float("inf")
    for item in _candidate_items(path):
        estimator.load(item.state_dict)
        correction = prediction_vector(estimator.predict(X_val_subset))
        combined = offset_val + correction
        current_validation_loss = float(validation_metric(y_val, combined))
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_item = item

    if best_item is None:
        best_item = path[0]
        estimator.load(best_item.state_dict)
        correction = prediction_vector(estimator.predict(X_val_subset))
        best_validation_loss = float(validation_metric(y_val, offset_val + correction))
    else:
        estimator.load(best_item.state_dict)

    model = FeatureSubsetModelAdapter(estimator, feature_indices, n_features)
    theta = theta_from_model(estimator, n_features, feature_indices)
    correction_support = support_from_model(
        estimator,
        n_features,
        support_tol=support_tol,
        feature_indices=feature_indices,
        prefer_input_mask=prefer_input_mask,
    )
    new_support, adjusted_support, final_support = decompose_support(
        common_support,
        correction_support,
    )
    return Stage2GroupResult(
        group=group,
        alpha=float(alpha),
        selected_lambda=float(best_item.lambda_),
        correction_model=model,
        correction_theta=theta,
        correction_support=correction_support,
        new_support=new_support,
        adjusted_support=adjusted_support,
        final_support=final_support,
        validation_mse=float(best_validation_loss),
        path=path,
        validation_loss=float(best_validation_loss),
    )
