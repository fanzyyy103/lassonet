from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lassonet import QuantileLassoNetRegressor

from twostage_lassonet_v2.utils import prediction_vector


@dataclass
class QuantileLassoNetBICResult:
    tau: float
    best_index: int
    best_lambda: float
    best_bic: float
    validation_pinball_at_selected_lambda: float
    n_selected: int
    support_mask: np.ndarray
    selected_features: list[str]
    prediction: np.ndarray
    path_rows: list[dict[str, Any]]
    criterion_name: str


def sigma_tau(tau: float) -> float:
    return float((1.0 - abs(1.0 - 2.0 * float(tau))) / 2.0)


def bic_proxy(
    *,
    tau: float,
    mean_pinball: float,
    n_selected: int,
    n_obs: int,
) -> float:
    pinball_sum = float(mean_pinball) * int(n_obs)
    return float((2.0 / sigma_tau(tau)) * pinball_sum + int(n_selected) * np.log(int(n_obs)))


def selected_feature_names_from_mask(
    support_mask: np.ndarray,
    feature_names: np.ndarray | list[str] | None = None,
) -> list[str]:
    mask = np.asarray(support_mask, dtype=bool)
    if feature_names is None:
        return [f"x{i + 1}" for i, keep in enumerate(mask) if keep]
    names = np.asarray(feature_names, dtype=object)
    return [str(name) for name, keep in zip(names, mask, strict=False) if keep]


def fit_quantile_lassonet_bic(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    tau: float,
    lassonet_config: dict[str, Any],
    feature_names: np.ndarray | list[str] | None = None,
) -> QuantileLassoNetBICResult:
    """
    Train Quantile LassoNet on a fixed lambda path and select lambda by a BIC-style proxy.

    BIC_tau(lambda) =
        (2 / sigma_tau) * sum_i rho_tau(y_i - qhat_i) + nu_lambda * log(n)

    In this implementation:
    - rho_tau is represented by the validation mean pinball loss times n
    - nu_lambda is the number of selected features from the LassoNet skip support
    - n is the validation sample size

    Notes:
    - This is the practical BIC-style proxy used in the current experiments.
    - nu_lambda here is not claimed to be the exact degrees of freedom.
    """
    print(f"[QLN-BIC effective config] tau={float(tau):g}, config={lassonet_config}", flush=True)
    model = QuantileLassoNetRegressor(tau=tau, **lassonet_config)
    path = model.path(
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        return_state_dicts=True,
    )

    if len(path) == 0:
        raise RuntimeError("Quantile LassoNet returned an empty lambda path.")

    bic_values: list[float] = []
    path_rows: list[dict[str, Any]] = []

    for index, item in enumerate(path):
        mask = np.asarray(item.selected, dtype=bool)
        mean_val_pinball = float(item.val_loss)
        n_selected = int(mask.sum())
        bic_value = bic_proxy(
            tau=tau,
            mean_pinball=mean_val_pinball,
            n_selected=n_selected,
            n_obs=len(y_val),
        )
        bic_values.append(bic_value)
        path_rows.append(
            {
                "tau": float(tau),
                "path_index": int(index),
                "lambda": float(item.lambda_),
                "validation_pinball_loss": mean_val_pinball,
                "n_selected": n_selected,
                "selected_features": selected_feature_names_from_mask(mask, feature_names),
                "fit_term": float((2.0 / sigma_tau(tau)) * mean_val_pinball * len(y_val)),
                "penalty_term": float(n_selected * np.log(len(y_val))),
                "penalty_ratio": float(
                    (n_selected * np.log(len(y_val)))
                    / ((2.0 / sigma_tau(tau)) * mean_val_pinball * len(y_val))
                )
                if mean_val_pinball > 0
                else np.nan,
                "BIC": float(bic_value),
                "training_iterations": int(item.n_iters),
            }
        )

    best_index = int(np.argmin(np.asarray(bic_values, dtype=np.float64)))
    best_item = path[best_index]
    model.load(best_item)

    support_mask = np.asarray(model.model.input_mask().detach().cpu().numpy(), dtype=bool)
    prediction = prediction_vector(model.predict(X_test.astype(np.float32))).astype(np.float64)
    selected_features = selected_feature_names_from_mask(support_mask, feature_names)

    for row_index, row in enumerate(path_rows):
        row["selected_by_BIC"] = row_index == best_index

    return QuantileLassoNetBICResult(
        tau=float(tau),
        best_index=best_index,
        best_lambda=float(best_item.lambda_),
        best_bic=float(bic_values[best_index]),
        validation_pinball_at_selected_lambda=float(best_item.val_loss),
        n_selected=int(support_mask.sum()),
        support_mask=support_mask,
        selected_features=selected_features,
        prediction=prediction,
        path_rows=path_rows,
        criterion_name=type(model.criterion).__name__,
    )
