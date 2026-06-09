from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .two_stage import TwoStagePretrainedLassoNetRegressor


@dataclass
class PathItem:
    lambda_: float
    score: float
    n_selected_features: int
    common_support: np.ndarray
    final_support_union: np.ndarray
    model: TwoStagePretrainedLassoNetRegressor


def mse_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def fit_two_stage_lambda_path(
    X_train,
    y_train,
    groups_train,
    *,
    lambda_seq: Sequence[float],
    alpha=0.5,
    stage1_lambda=1e-2,
    stage1_M=10.0,
    stage2_M=10.0,
    common_model_kwargs=None,
    group_model_kwargs=None,
    X_eval=None,
    y_eval=None,
    groups_eval=None,
    X_val=None,
    y_val=None,
    groups_val=None,
    score_fn: Optional[Callable] = None,
):
    if score_fn is None:
        score_fn = mse_score

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    groups_train = np.asarray(groups_train)

    if X_eval is None:
        X_eval = X_train
        y_eval = y_train
        groups_eval = groups_train
    else:
        X_eval = np.asarray(X_eval, dtype=np.float32)
        y_eval = np.asarray(y_eval, dtype=np.float32)
        groups_eval = np.asarray(groups_eval)

    items = []
    for current_lambda in lambda_seq:
        model = TwoStagePretrainedLassoNetRegressor(
            alpha=alpha,
            stage1_lambda=stage1_lambda,
            stage2_lambda=float(current_lambda),
            stage1_M=stage1_M,
            stage2_M=stage2_M,
            common_model_kwargs=deepcopy(common_model_kwargs or {}),
            group_model_kwargs=deepcopy(group_model_kwargs or {}),
        )
        model.fit(
            X_train,
            y_train,
            groups_train,
            X_val=X_val,
            y_val=y_val,
            groups_val=groups_val,
        )

        y_pred = model.predict(X_eval, groups_eval)
        score = float(score_fn(y_eval, y_pred))

        common_support = model.get_common_support()
        final_union = np.array([], dtype=int)
        for group in model.group_order_:
            final_union = np.union1d(final_union, model.get_group_final_support(group))

        items.append(
            PathItem(
                lambda_=float(current_lambda),
                score=score,
                n_selected_features=int(final_union.size),
                common_support=common_support,
                final_support_union=final_union,
                model=model,
            )
        )

    return items
