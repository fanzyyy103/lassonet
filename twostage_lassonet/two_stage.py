from copy import deepcopy

import numpy as np

from .regression import LassoNetRegressor


def _to_numpy_groups(groups):
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError("groups must be a 1D array-like object.")
    return group_array


class TwoStagePretrainedLassoNetRegressor:
    """
    Two-stage pretrained LassoNet for grouped samples.

    Stage 1:
        fit a common LassoNet on all observations.

    Stage 2:
        for each sample group k, fit a group-specific LassoNet on
        y_k - (1 - alpha) * f0_hat(X_k)
        with feature-wise penalty factor
            pf_j = 1                      if j in S0_hat
            pf_j = 1 / alpha             otherwise
        and the same LassoNet hierarchy constraint.
    """

    def __init__(
        self,
        *,
        alpha=0.5,
        stage1_lambda=1e-2,
        stage2_lambda=1e-2,
        stage1_M=10.0,
        stage2_M=10.0,
        common_model_kwargs=None,
        group_model_kwargs=None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

        self.alpha = float(alpha)
        self.stage1_lambda = stage1_lambda
        self.stage2_lambda = stage2_lambda
        self.stage1_M = stage1_M
        self.stage2_M = stage2_M
        self.common_model_kwargs = common_model_kwargs or {}
        self.group_model_kwargs = group_model_kwargs or {}

        self.common_model_ = None
        self.group_models_ = {}
        self.group_order_ = None
        self.common_support_ = None
        self.penalty_factor_ = None

    def _resolve_group_value(self, value, group):
        if isinstance(value, dict):
            if group not in value:
                raise KeyError(f"Missing group-specific value for group {group!r}.")
            return value[group]
        return value

    def _build_penalty_factor(self, common_support):
        if self.alpha == 0.0:
            penalty_factor = np.ones(common_support.shape[0], dtype=np.float32)
            zero_mask = ~common_support
        else:
            penalty_factor = np.full(
                common_support.shape[0],
                1.0 / self.alpha,
                dtype=np.float32,
            )
            penalty_factor[common_support] = 1.0
            zero_mask = np.zeros(common_support.shape[0], dtype=bool)
        return penalty_factor, zero_mask

    def fit(
        self,
        X,
        y,
        groups,
        *,
        X_val=None,
        y_val=None,
        groups_val=None,
    ):
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        groups_array = _to_numpy_groups(groups)

        if X_array.shape[0] != groups_array.shape[0]:
            raise ValueError("X and groups must have the same number of rows.")
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.group_order_ = np.unique(groups_array)

        self.common_model_ = LassoNetRegressor(**deepcopy(self.common_model_kwargs))
        self.common_model_.fit(
            X_array,
            y_array,
            lambda_=self.stage1_lambda,
            M=self.stage1_M,
            X_val=X_val,
            y_val=y_val,
        )
        self.common_support_ = self.common_model_.selected_mask()
        self.penalty_factor_, zero_mask = self._build_penalty_factor(self.common_support_)

        common_predictions = self.common_model_.predict(X_array)
        common_predictions = np.asarray(common_predictions, dtype=np.float32)
        offset = (1.0 - self.alpha) * common_predictions

        if X_val is not None and y_val is not None:
            X_val_array = np.asarray(X_val, dtype=np.float32)
            y_val_array = np.asarray(y_val, dtype=np.float32)
            if groups_val is None:
                raise ValueError("groups_val must be supplied when X_val and y_val are supplied.")
            groups_val_array = _to_numpy_groups(groups_val)
            common_predictions_val = self.common_model_.predict(X_val_array)
            common_predictions_val = np.asarray(common_predictions_val, dtype=np.float32)
            offset_val = (1.0 - self.alpha) * common_predictions_val
        else:
            X_val_array = y_val_array = groups_val_array = offset_val = None

        self.group_models_ = {}
        for group in self.group_order_:
            train_mask = groups_array == group
            group_y = y_array[train_mask] - offset[train_mask]

            X_group_val = None
            y_group_val = None
            if X_val_array is not None:
                val_mask = groups_val_array == group
                if val_mask.any():
                    X_group_val = X_val_array[val_mask]
                    y_group_val = y_val_array[val_mask] - offset_val[val_mask]

            group_model = LassoNetRegressor(**deepcopy(self.group_model_kwargs))
            group_model.fit(
                X_array[train_mask],
                group_y,
                lambda_=self._resolve_group_value(self.stage2_lambda, group),
                penalty_factor=self.penalty_factor_,
                M=self._resolve_group_value(self.stage2_M, group),
                zero_mask=zero_mask,
                X_val=X_group_val,
                y_val=y_group_val,
            )
            self.group_models_[group] = group_model

        return self

    def predict_components(self, X, groups):
        if self.common_model_ is None:
            raise RuntimeError("fit must be called before predict_components.")

        X_array = np.asarray(X, dtype=np.float32)
        groups_array = _to_numpy_groups(groups)
        if X_array.shape[0] != groups_array.shape[0]:
            raise ValueError("X and groups must have the same number of rows.")

        common_component = (1.0 - self.alpha) * np.asarray(
            self.common_model_.predict(X_array),
            dtype=np.float32,
        )
        group_component = np.zeros_like(common_component)

        for group, model in self.group_models_.items():
            mask = groups_array == group
            if not mask.any():
                continue
            group_component[mask] = np.asarray(
                model.predict(X_array[mask]),
                dtype=np.float32,
            )

        return {
            "common": common_component,
            "group_specific": group_component,
            "final": common_component + group_component,
        }

    def predict(self, X, groups):
        return self.predict_components(X, groups)["final"]

    def get_common_support(self):
        return np.flatnonzero(self.common_support_)

    def get_group_support(self, group):
        return np.flatnonzero(self.group_models_[group].selected_mask())

    def get_group_individual_support(self, group):
        group_support = self.group_models_[group].selected_mask()
        return np.flatnonzero(group_support & ~self.common_support_)

    def get_group_final_support(self, group):
        group_support = self.group_models_[group].selected_mask()
        return np.flatnonzero(group_support | self.common_support_)
