from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def as_2d_float_array(X):
    if hasattr(X, "to_numpy"):
        array = X.to_numpy()
    else:
        array = np.asarray(X)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("X must be a 2D array or pandas DataFrame.")
    return array


def as_1d_float_array(y, *, name: str = "y"):
    array = np.asarray(y, dtype=np.float32).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def as_1d_group_array(groups):
    array = np.asarray(groups)
    if array.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    return array


def first_seen_unique(values: Sequence[Any]):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return np.asarray(seen, dtype=object)


def resolve_feature_names(X, n_features: int):
    if hasattr(X, "columns"):
        return np.asarray([str(col) for col in X.columns], dtype=object)
    return np.asarray([f"x{i}" for i in range(n_features)], dtype=object)


def validate_feature_names(X, feature_names):
    if hasattr(X, "columns"):
        incoming = np.asarray([str(col) for col in X.columns], dtype=object)
        if incoming.shape != feature_names.shape or np.any(incoming != feature_names):
            raise ValueError("Prediction DataFrame columns must match the fitted columns.")


def prediction_vector(predictions):
    array = np.asarray(predictions, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError("Expected one-dimensional regression predictions.")
    return array


def theta_from_model(model, n_features: int, feature_indices: Iterable[int] | None = None):
    theta = np.zeros(n_features, dtype=np.float64)
    if getattr(model, "model", None) is None:
        return theta
    weights = model.model.skip.weight.detach().norm(p=2, dim=0).cpu().numpy()
    if feature_indices is None:
        if weights.shape[0] != n_features:
            raise ValueError("Model theta width does not match n_features.")
        theta[:] = weights
    else:
        feature_indices = np.asarray(feature_indices, dtype=int)
        theta[feature_indices] = weights
    return theta


def support_from_theta(theta, support_tol: float):
    return np.asarray(np.abs(theta) > support_tol, dtype=bool)


def support_from_model(
    model,
    n_features: int,
    *,
    support_tol: float,
    feature_indices: Iterable[int] | None = None,
    prefer_input_mask: bool = True,
):
    if prefer_input_mask and hasattr(model, "selected_mask") and feature_indices is None:
        selected = model.selected_mask()
    elif prefer_input_mask and getattr(model, "best_selected_", None) is not None and feature_indices is None:
        selected = model.best_selected_
    elif prefer_input_mask and getattr(model, "model", None) is not None and feature_indices is None:
        selected = model.model.input_mask()
    else:
        selected = None

    if selected is not None:
        if hasattr(selected, "detach"):
            selected = selected.detach().cpu().numpy()
        selected = np.asarray(selected, dtype=bool)
        if selected.shape == (n_features,):
            return selected

    theta = theta_from_model(model, n_features, feature_indices)
    return support_from_theta(theta, support_tol)


def mask_to_indices(mask):
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0.0:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_pinball_loss(y_true, y_pred, tau: float):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching one-dimensional shapes.")
    residual = y_true - y_pred
    loss = np.maximum(tau * residual, (tau - 1.0) * residual)
    return float(loss.mean())


def empirical_coverage(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching one-dimensional shapes.")
    return float(np.mean(y_true <= y_pred))


def support_precision_recall_f1(selected_mask, true_mask):
    selected_mask = np.asarray(selected_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    if selected_mask.shape != true_mask.shape:
        raise ValueError("selected_mask and true_mask must have the same shape.")
    true_positive = int(np.sum(selected_mask & true_mask))
    false_positive = int(np.sum(selected_mask & ~true_mask))
    false_negative = int(np.sum(~selected_mask & true_mask))
    precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
