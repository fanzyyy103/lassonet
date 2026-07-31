from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def resolve_alpha_values(alpha, alpha_values=None):
    if alpha == "auto":
        values = alpha_values if alpha_values is not None else np.linspace(0.0, 1.0, 11)
    else:
        values = [float(alpha)]
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("alpha_values must be a non-empty one-dimensional sequence.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("All alpha values must be in [0, 1].")
    return values


def resolve_lambda_sequence(lambda_spec, *, group=None):
    if isinstance(lambda_spec, Mapping):
        if group not in lambda_spec:
            raise KeyError(f"Missing lambda sequence/value for group {group!r}.")
        lambda_spec = lambda_spec[group]
    if isinstance(lambda_spec, str) and lambda_spec == "auto":
        return None
    if np.isscalar(lambda_spec):
        return [float(lambda_spec)]
    values = [float(value) for value in lambda_spec]
    if not values:
        raise ValueError("Lambda sequence must not be empty.")
    if any(value < 0.0 for value in values):
        raise ValueError("Lambda values must be nonnegative.")
    return values


def build_penalty_factor(alpha: float, common_support):
    common_support = np.asarray(common_support, dtype=bool)
    if alpha == 0.0:
        return np.ones(common_support.shape[0], dtype=np.float32)
    penalty_factor = np.full(common_support.shape[0], 1.0 / alpha, dtype=np.float32)
    penalty_factor[common_support] = 1.0
    if not np.all(np.isfinite(penalty_factor)):
        raise ValueError("Penalty factors must be finite; alpha=0 is handled by feature masking.")
    return penalty_factor


def allowed_feature_indices(alpha: float, common_support):
    common_support = np.asarray(common_support, dtype=bool)
    if alpha == 0.0:
        return np.flatnonzero(common_support)
    return np.arange(common_support.shape[0], dtype=int)


def decompose_support(common_support, correction_support):
    common_support = np.asarray(common_support, dtype=bool)
    correction_support = np.asarray(correction_support, dtype=bool)
    if common_support.shape != correction_support.shape:
        raise ValueError("Support masks must have the same shape.")
    new_support = correction_support & ~common_support
    adjusted_support = correction_support & common_support
    final_support = common_support | correction_support
    return new_support, adjusted_support, final_support
