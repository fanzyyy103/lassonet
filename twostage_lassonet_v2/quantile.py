from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from lassonet import QuantileLassoNetRegressor

from .regressor import TwoStageLassoNetRegressorV2
from .results import AlphaCandidateResult, PredictionComponents, TwoStageFitResult
from .selection import (
    allowed_feature_indices,
    build_penalty_factor,
    resolve_alpha_values,
    resolve_lambda_sequence,
)
from .stage import fit_stage1, fit_stage2_group
from .utils import (
    empirical_coverage,
    first_seen_unique,
    mask_to_indices,
    mean_pinball_loss,
    prediction_vector,
    resolve_feature_names,
    support_precision_recall_f1,
    validate_feature_names,
)


def _validate_tau(tau: float) -> float:
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1.")
    return tau


class TwoStageQuantileLassoNetRegressor(TwoStageLassoNetRegressorV2):
    """Two-stage grouped Quantile LassoNet using the same tau in both stages."""

    def __init__(
        self,
        *,
        tau: float = 0.5,
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
        self.tau = _validate_tau(tau)
        super().__init__(
            hidden_dims_stage1=hidden_dims_stage1,
            hidden_dims_stage2=hidden_dims_stage2,
            M_stage1=M_stage1,
            M_stage2=M_stage2,
            alpha=alpha,
            alpha_values=alpha_values,
            lambda_stage1=lambda_stage1,
            lambda_stage2=lambda_stage2,
            support_tol=support_tol,
            validation_fraction=validation_fraction,
            random_state=random_state,
            refit=refit,
            verbose=verbose,
            stage1_model_kwargs=stage1_model_kwargs,
            stage2_model_kwargs=stage2_model_kwargs,
            **lassonet_parameters,
        )

    def _stage_kwargs(self, explicit_kwargs):
        kwargs = super()._stage_kwargs(explicit_kwargs)
        incoming_tau = kwargs.pop("tau", self.tau)
        if float(incoming_tau) != self.tau:
            raise ValueError("Stage 1 and Stage 2 must use the same tau as the estimator.")
        kwargs["tau"] = self.tau
        return kwargs

    def _pinball_loss(self, y_true, y_pred) -> float:
        return mean_pinball_loss(y_true, y_pred, self.tau)

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
            estimator_cls=QuantileLassoNetRegressor,
            prefer_input_mask=False,
        )

        stage1_train_prediction = prediction_vector(stage1.model.predict(X_train))
        stage1_val_prediction = prediction_vector(stage1.model.predict(X_val))

        alpha_results = []
        for alpha_index, alpha_value in enumerate(self.alpha_values_):
            alpha_value = float(alpha_value)
            self._log(f"[Two-Stage Quantile] fitting Stage 2 candidates for alpha={alpha_value:.3g}")
            penalty_factor = build_penalty_factor(alpha_value, stage1.support)
            feature_indices = allowed_feature_indices(alpha_value, stage1.support)
            stage2_lambda_seq = [] if feature_indices.size == 0 else None

            group_results = {}
            group_predictions = np.full(X_val.shape[0], np.nan, dtype=np.float32)
            group_losses = {}
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
                    estimator_cls=QuantileLassoNetRegressor,
                    validation_metric=self._pinball_loss,
                    prefer_input_mask=False,
                )
                correction = prediction_vector(result.correction_model.predict(X_val[val_mask]))
                group_predictions[val_mask] = offset_val + correction
                group_results[group] = result
                group_losses[group] = self._pinball_loss(y_val[val_mask], group_predictions[val_mask])
                group_lambdas[group] = result.selected_lambda
                n_selected[group] = int(result.final_support.sum())

            if np.isnan(group_predictions).any():
                raise RuntimeError("Some validation predictions were not filled.")

            validation_loss = self._pinball_loss(y_val, group_predictions)
            alpha_results.append(
                AlphaCandidateResult(
                    alpha=alpha_value,
                    validation_mse=validation_loss,
                    group_validation_mse=group_losses,
                    group_lambdas=group_lambdas,
                    group_results=group_results,
                    n_selected_features_by_group=n_selected,
                    validation_loss=validation_loss,
                    group_validation_loss=group_losses,
                )
            )

        selected_alpha_result = min(
            alpha_results,
            key=lambda item: item.validation_loss if item.validation_loss is not None else item.validation_mse,
        )
        return TwoStageFitResult(
            stage1=stage1,
            alpha_results=alpha_results,
            selected_alpha=selected_alpha_result.alpha,
            selected_stage2=selected_alpha_result.group_results,
        )

    def fit(self, X, y, groups, X_val=None, y_val=None, groups_val=None):
        super().fit(X, y, groups, X_val=X_val, y_val=y_val, groups_val=groups_val)
        self.stage1_validation_losses_ = self.stage1_validation_scores_
        self.alpha_validation_losses_ = {
            result.alpha: (result.validation_loss if result.validation_loss is not None else result.validation_mse)
            for result in self.tuning_results_
        }
        self.stage2_supports_ = self.group_supports_
        self.stage2_new_supports_ = self.new_supports_
        self.stage2_adjusted_supports_ = self.adjusted_supports_
        self.stage2_final_supports_ = self.final_supports_
        self.lambda_stage1_ = self.stage1_lambda_
        self.lambda_stage2_ = self.group_lambdas_
        return self

    def score(self, X, y, groups):
        return -self.pinball_loss(X, y, groups)

    def pinball_loss(self, X, y, groups):
        prediction = self.predict(X, groups)
        return self._pinball_loss(y, prediction)

    def pinball_loss_by_group(self, X, y, groups):
        prediction = self.predict(X, groups)
        groups_array = np.asarray(groups)
        y_array = np.asarray(y, dtype=np.float64).reshape(-1)
        results = {}
        for group in self.groups_:
            mask = groups_array == group
            if mask.any():
                results[group] = self._pinball_loss(y_array[mask], prediction[mask])
        return results

    def coverage(self, X, y, groups):
        prediction = self.predict(X, groups)
        return empirical_coverage(y, prediction)

    def coverage_by_group(self, X, y, groups):
        prediction = self.predict(X, groups)
        groups_array = np.asarray(groups)
        y_array = np.asarray(y, dtype=np.float64).reshape(-1)
        results = {}
        for group in self.groups_:
            mask = groups_array == group
            if mask.any():
                results[group] = empirical_coverage(y_array[mask], prediction[mask])
        return results

    def get_stage1_support(self):
        return self.get_common_support()

    def get_stage2_support(self, group):
        return self.get_correction_support(group)

    def get_group_supports(self, *, strict_new: bool = False):
        if strict_new:
            return {group: self.get_new_support(group) for group in self.groups_}
        return {group: self.get_stage2_support(group) for group in self.groups_}

    def get_tuning_summary(self):
        self._check_is_fitted()
        rows = []
        for result in self.tuning_results_:
            validation_loss = result.validation_loss if result.validation_loss is not None else result.validation_mse
            group_loss_map = result.group_validation_loss or result.group_validation_mse
            for group in self.groups_:
                rows.append(
                    {
                        "alpha": result.alpha,
                        "selected": result.alpha == self.alpha_,
                        "validation_pinball_loss": validation_loss,
                        "group": group,
                        "group_validation_pinball_loss": group_loss_map[group],
                        "stage2_lambda": result.group_lambdas[group],
                        "n_final_selected": result.n_selected_features_by_group[group],
                    }
                )
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def support_metrics(self, selected_mask, true_mask):
        return support_precision_recall_f1(selected_mask, true_mask)
