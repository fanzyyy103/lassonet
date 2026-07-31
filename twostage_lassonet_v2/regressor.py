from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import numpy as np
from sklearn.model_selection import train_test_split

from .results import AlphaCandidateResult, PredictionComponents, TwoStageFitResult
from .selection import (
    allowed_feature_indices,
    build_penalty_factor,
    resolve_alpha_values,
    resolve_lambda_sequence,
)
from .stage import fit_stage1, fit_stage2_group
from .utils import (
    as_1d_float_array,
    as_1d_group_array,
    as_2d_float_array,
    first_seen_unique,
    mask_to_indices,
    mse,
    prediction_vector,
    r2_score,
    resolve_feature_names,
    validate_feature_names,
)


class TwoStageLassoNetRegressorV2:
    """
    Two-stage pretrained LassoNet regressor following the explicit
    common-plus-correction formulation.
    """

    def __init__(
        self,
        *,
        hidden_dims_stage1=(64, 32),
        hidden_dims_stage2=(16,),
        M_stage1=10.0,
        M_stage2=10.0,
        alpha="auto",
        alpha_values=None,
        lambda_stage1="auto",
        lambda_stage2="auto",
        support_tol=1e-8,
        validation_fraction=0.2,
        random_state=42,
        refit=True,
        verbose=1,
        stage1_model_kwargs=None,
        stage2_model_kwargs=None,
        **lassonet_parameters,
    ):
        self.hidden_dims_stage1 = tuple(hidden_dims_stage1)
        self.hidden_dims_stage2 = tuple(hidden_dims_stage2)
        self.M_stage1 = M_stage1
        self.M_stage2 = M_stage2
        self.alpha = alpha
        self.alpha_values = alpha_values
        self.lambda_stage1 = lambda_stage1
        self.lambda_stage2 = lambda_stage2
        self.support_tol = support_tol
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.refit = refit
        self.verbose = verbose
        self.stage1_model_kwargs = stage1_model_kwargs
        self.stage2_model_kwargs = stage2_model_kwargs
        self.lassonet_parameters = lassonet_parameters

        self.is_fitted_ = False

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _stage_kwargs(self, explicit_kwargs):
        kwargs = deepcopy(self.lassonet_parameters)
        kwargs.update(deepcopy(explicit_kwargs or {}))
        if float(kwargs.get("gamma", 0.0)) != 0.0 or float(kwargs.get("gamma_skip", 0.0)) != 0.0:
            raise ValueError("gamma and gamma_skip must remain 0.0; V2 does not use Ridge/L2 penalties.")
        kwargs.setdefault("gamma", 0.0)
        kwargs.setdefault("gamma_skip", 0.0)
        return kwargs

    def _validate_fit_inputs(self, X, y, groups):
        X_array = as_2d_float_array(X)
        y_array = as_1d_float_array(y)
        groups_array = as_1d_group_array(groups)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have matching lengths.")
        if X_array.shape[0] != groups_array.shape[0]:
            raise ValueError("X and groups must have matching lengths.")
        if X_array.shape[0] < 2:
            raise ValueError("At least two samples are required.")
        return X_array, y_array, groups_array

    def _split_train_validation(self, X, y, groups, X_val, y_val, groups_val):
        if X_val is not None or y_val is not None or groups_val is not None:
            if X_val is None or y_val is None or groups_val is None:
                raise ValueError("X_val, y_val, and groups_val must be supplied together.")
            X_val_array, y_val_array, groups_val_array = self._validate_fit_inputs(
                X_val,
                y_val,
                groups_val,
            )
            return X, y, groups, X_val_array, y_val_array, groups_val_array

        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1) when validation data is not supplied.")
        unique, counts = np.unique(groups, return_counts=True)
        if np.any(counts < 2):
            sparse = unique[counts < 2].tolist()
            raise ValueError(f"Each group needs at least two samples for stratified validation split: {sparse}.")
        indices = np.arange(X.shape[0])
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=groups,
        )
        return (
            X[train_idx],
            y[train_idx],
            groups[train_idx],
            X[val_idx],
            y[val_idx],
            groups[val_idx],
        )

    def _fit_pipeline(self, X_train, y_train, groups_train, X_val, y_val, groups_val, *, refit_mode=False):
        stage1_lambda_seq = resolve_lambda_sequence(self.lambda_stage1)
        stage1_seed = int(self.random_state) + (20000 if refit_mode else 101)
        stage1 = fit_stage1(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            lambda_seq=stage1_lambda_seq,
            hidden_dims=self.hidden_dims_stage1,
            M=self.M_stage1,
            support_tol=self.support_tol,
            model_kwargs=self._stage_kwargs(self.stage1_model_kwargs),
            seed=stage1_seed,
            verbose=0 if self.verbose <= 1 else self.verbose,
        )

        stage1_train_prediction = prediction_vector(stage1.model.predict(X_train))
        stage1_val_prediction = prediction_vector(stage1.model.predict(X_val))

        alpha_results = []
        for alpha_index, alpha_value in enumerate(self.alpha_values_):
            alpha_value = float(alpha_value)
            self._log(f"[V2] fitting Stage 2 candidates for alpha={alpha_value:.3g}")
            penalty_factor = build_penalty_factor(alpha_value, stage1.support)
            feature_indices = allowed_feature_indices(alpha_value, stage1.support)
            if feature_indices.size == 0:
                stage2_lambda_seq = []
            else:
                stage2_lambda_seq = None

            group_results = {}
            group_predictions = np.full(X_val.shape[0], np.nan, dtype=np.float32)
            group_mses = {}
            group_lambdas = {}
            n_selected = {}

            for group_index, group in enumerate(self.groups_):
                train_mask = groups_train == group
                val_mask = groups_val == group
                if not train_mask.any():
                    raise ValueError(f"No training observations for group {group!r}.")
                if not val_mask.any():
                    raise ValueError(f"No validation observations for group {group!r}.")

                lambda_seq = resolve_lambda_sequence(self.lambda_stage2, group=group)
                if feature_indices.size == 0 and lambda_seq is None:
                    lambda_seq = []
                if stage2_lambda_seq == []:
                    lambda_seq = []

                offset_train = (1.0 - alpha_value) * stage1_train_prediction[train_mask]
                offset_val = (1.0 - alpha_value) * stage1_val_prediction[val_mask]
                result = fit_stage2_group(
                    X_train[train_mask],
                    y_train[train_mask],
                    X_val=X_val[val_mask],
                    y_val=y_val[val_mask],
                    offset_train=offset_train,
                    offset_val=offset_val,
                    group=group,
                    alpha=alpha_value,
                    common_support=stage1.support,
                    feature_indices=feature_indices,
                    penalty_factor=penalty_factor,
                    lambda_seq=lambda_seq,
                    hidden_dims=self.hidden_dims_stage2,
                    M=self.M_stage2,
                    support_tol=self.support_tol,
                    model_kwargs=self._stage_kwargs(self.stage2_model_kwargs),
                    seed=int(self.random_state) + (30000 if refit_mode else 1000) + alpha_index * 100 + group_index,
                    verbose=0 if self.verbose <= 1 else self.verbose,
                )
                correction = prediction_vector(result.correction_model.predict(X_val[val_mask]))
                group_predictions[val_mask] = offset_val + correction
                group_results[group] = result
                group_mses[group] = mse(y_val[val_mask], group_predictions[val_mask])
                group_lambdas[group] = result.selected_lambda
                n_selected[group] = int(result.final_support.sum())

            if np.isnan(group_predictions).any():
                raise RuntimeError("Some validation predictions were not filled.")
            alpha_results.append(
                AlphaCandidateResult(
                    alpha=alpha_value,
                    validation_mse=mse(y_val, group_predictions),
                    group_validation_mse=group_mses,
                    group_lambdas=group_lambdas,
                    group_results=group_results,
                    n_selected_features_by_group=n_selected,
                )
            )

        selected_alpha_result = min(alpha_results, key=lambda item: item.validation_mse)
        return TwoStageFitResult(
            stage1=stage1,
            alpha_results=alpha_results,
            selected_alpha=selected_alpha_result.alpha,
            selected_stage2=selected_alpha_result.group_results,
        )

    def fit(self, X, y, groups, X_val=None, y_val=None, groups_val=None):
        X_array, y_array, groups_array = self._validate_fit_inputs(X, y, groups)
        self.feature_names_in_ = resolve_feature_names(X, X_array.shape[1])
        self.n_features_in_ = X_array.shape[1]
        self.groups_ = first_seen_unique(groups_array)
        self.alpha_values_ = resolve_alpha_values(self.alpha, self.alpha_values)

        split = self._split_train_validation(X_array, y_array, groups_array, X_val, y_val, groups_val)
        X_train, y_train, groups_train, X_valid, y_valid, groups_valid = split

        self._log("[V2] fitting pooled Stage 1 once")
        fit_result = self._fit_pipeline(
            X_train,
            y_train,
            groups_train,
            X_valid,
            y_valid,
            groups_valid,
            refit_mode=False,
        )
        selected_alpha = fit_result.selected_alpha

        if self.refit:
            original_alpha_values = self.alpha_values_
            original_lambda_stage1 = self.lambda_stage1
            original_lambda_stage2 = self.lambda_stage2
            try:
                self.alpha_values_ = np.asarray([selected_alpha], dtype=np.float64)
                self.lambda_stage1 = fit_result.stage1.selected_lambda
                selected_lambdas = {
                    group: result.selected_lambda
                    for group, result in fit_result.selected_stage2.items()
                }
                self.lambda_stage2 = selected_lambdas
                refit_X = np.vstack([X_train, X_valid])
                refit_y = np.concatenate([y_train, y_valid])
                refit_groups = np.concatenate([groups_train, groups_valid])
                refit_result = self._fit_pipeline(
                    refit_X,
                    refit_y,
                    refit_groups,
                    refit_X,
                    refit_y,
                    refit_groups,
                    refit_mode=True,
                )
            finally:
                self.alpha_values_ = original_alpha_values
                self.lambda_stage1 = original_lambda_stage1
                self.lambda_stage2 = original_lambda_stage2

            final_stage1 = refit_result.stage1
            final_stage2 = refit_result.selected_stage2
        else:
            final_stage1 = fit_result.stage1
            final_stage2 = fit_result.selected_stage2

        self.fit_result_ = fit_result
        self.stage1_model_ = final_stage1.model
        self.stage1_lambda_ = final_stage1.selected_lambda
        self.stage1_theta_ = final_stage1.theta
        self.stage1_support_ = final_stage1.support
        self.stage1_path_ = fit_result.stage1.path
        self.stage1_validation_scores_ = fit_result.stage1.validation_scores
        self.alpha_ = selected_alpha
        self.alpha_validation_scores_ = {
            result.alpha: result.validation_mse for result in fit_result.alpha_results
        }
        self.tuning_results_ = fit_result.alpha_results
        self.group_models_ = {
            group: result.correction_model for group, result in final_stage2.items()
        }
        self.group_lambdas_ = {
            group: result.selected_lambda for group, result in final_stage2.items()
        }
        self.group_supports_ = {
            group: result.correction_support for group, result in final_stage2.items()
        }
        self.new_supports_ = {
            group: result.new_support for group, result in final_stage2.items()
        }
        self.adjusted_supports_ = {
            group: result.adjusted_support for group, result in final_stage2.items()
        }
        self.final_supports_ = {
            group: result.final_support for group, result in final_stage2.items()
        }
        self.group_results_ = final_stage2
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("fit must be called before this method.")

    def _prepare_predict_inputs(self, X, groups=None):
        self._check_is_fitted()
        validate_feature_names(X, self.feature_names_in_)
        X_array = as_2d_float_array(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features used during fit.")
        if groups is None:
            return X_array, None
        groups_array = as_1d_group_array(groups)
        if groups_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and groups must have matching lengths.")
        unknown = sorted(set(groups_array.tolist()) - set(self.groups_.tolist()))
        if unknown:
            raise ValueError(f"Unknown group label(s) at prediction time: {unknown}.")
        return X_array, groups_array

    def predict_common(self, X):
        X_array, _ = self._prepare_predict_inputs(X)
        return prediction_vector(self.stage1_model_.predict(X_array))

    def predict_offset(self, X, alpha=None):
        alpha_value = self.alpha_ if alpha is None else float(alpha)
        return (1.0 - alpha_value) * self.predict_common(X)

    def predict_correction(self, X, groups):
        X_array, groups_array = self._prepare_predict_inputs(X, groups)
        correction = np.empty(X_array.shape[0], dtype=np.float32)
        for group in self.groups_:
            mask = groups_array == group
            if mask.any():
                correction[mask] = prediction_vector(self.group_models_[group].predict(X_array[mask]))
        return correction

    def predict_components(self, X, groups):
        common = self.predict_common(X)
        offset = (1.0 - self.alpha_) * common
        correction = self.predict_correction(X, groups)
        final = offset + correction
        return PredictionComponents(
            common_prediction=common,
            scaled_common_offset=offset,
            correction_prediction=correction,
            final_prediction=final,
        )

    def predict(self, X, groups):
        return self.predict_components(X, groups).final_prediction

    def score(self, X, y, groups):
        return r2_score(y, self.predict(X, groups))

    def _names_from_mask(self, mask):
        self._check_is_fitted()
        return self.feature_names_in_[np.asarray(mask, dtype=bool)]

    def get_common_support(self):
        return self._names_from_mask(self.stage1_support_)

    def get_correction_support(self, group):
        return self._names_from_mask(self.group_supports_[group])

    def get_new_support(self, group):
        return self._names_from_mask(self.new_supports_[group])

    def get_adjusted_support(self, group):
        return self._names_from_mask(self.adjusted_supports_[group])

    def get_final_support(self, group):
        return self._names_from_mask(self.final_supports_[group])

    def get_support_summary(self):
        self._check_is_fitted()
        rows = []
        for group in self.groups_:
            stage2 = self.group_supports_[group]
            for index, feature in enumerate(self.feature_names_in_):
                in_stage1 = bool(self.stage1_support_[index])
                in_stage2 = bool(stage2[index])
                if in_stage1 and in_stage2:
                    support_type = "adjusted_common"
                elif in_stage1:
                    support_type = "common_only"
                elif in_stage2:
                    support_type = "new_group_specific"
                else:
                    support_type = "not_selected"
                rows.append(
                    {
                        "feature": feature,
                        "stage1_selected": in_stage1,
                        "group": group,
                        "stage2_selected": in_stage2,
                        "support_type": support_type,
                    }
                )
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def get_tuning_summary(self):
        self._check_is_fitted()
        rows = []
        for result in self.tuning_results_:
            for group in self.groups_:
                rows.append(
                    {
                        "alpha": result.alpha,
                        "selected": result.alpha == self.alpha_,
                        "validation_mse": result.validation_mse,
                        "group": group,
                        "group_validation_mse": result.group_validation_mse[group],
                        "stage2_lambda": result.group_lambdas[group],
                        "n_final_selected": result.n_selected_features_by_group[group],
                    }
                )
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def get_common_support_indices(self):
        return mask_to_indices(self.stage1_support_)

    def get_correction_support_indices(self, group):
        return mask_to_indices(self.group_supports_[group])

    def get_new_support_indices(self, group):
        return mask_to_indices(self.new_supports_[group])

    def get_adjusted_support_indices(self, group):
        return mask_to_indices(self.adjusted_supports_[group])

    def get_final_support_indices(self, group):
        return mask_to_indices(self.final_supports_[group])
