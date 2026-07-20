from copy import deepcopy
import inspect
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, check_cv

from lassonet import LassoNetRegressor, LassoNetRegressorCV

from .ptlasso_oriented import (
    PTLassoOrientedTwoStageLassoNetRegressor,
    _resolve_feature_names,
)
from .two_stage import (
    _build_penalty_factor,
    _selected_mask_from_model,
    _to_numpy_groups,
    _to_prediction_vector,
    extract_stage1_artifacts,
)


def _translate_cv_kwargs(model_kwargs, *, default_M):
    """Accept our legacy epoch names while constructing official CV estimators."""
    kwargs = deepcopy(model_kwargs or {})
    kwargs.setdefault("M", default_M)

    if "dense_epochs" in kwargs or "sparse_epochs" in kwargs:
        dense_epochs = kwargs.pop("dense_epochs", 1000)
        sparse_epochs = kwargs.pop("sparse_epochs", 100)
        kwargs.setdefault("n_iters", (dense_epochs, sparse_epochs))

    if "dense_patience" in kwargs or "sparse_patience" in kwargs:
        dense_patience = kwargs.pop("dense_patience", 100)
        sparse_patience = kwargs.pop("sparse_patience", 10)
        kwargs.setdefault("patience", (dense_patience, sparse_patience))

    if kwargs.pop("skip_only", False):
        raise ValueError(
            "Official LassoNetRegressorCV does not implement skip_only. "
            "Stage 2 in this estimator is a full official LassoNet model."
        )

    if float(kwargs.get("gamma", 0.0)) != 0.0 or float(kwargs.get("gamma_skip", 0.0)) != 0.0:
        raise ValueError("gamma and gamma_skip must remain 0: this estimator does not use Ridge/L2 penalties.")

    return kwargs


def _official_supports_penalty_factor():
    return "penalty_factor" in inspect.signature(LassoNetRegressor).parameters


class _OfficialCVModelAdapter:
    """Map an official CV model trained on a feature subset back to full width."""

    def __init__(self, estimator, feature_indices, n_features):
        self.estimator = estimator
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.n_features = int(n_features)

    @property
    def best_lambda_(self):
        return float(self.estimator.best_lambda_)

    @property
    def best_cv_score_(self):
        return float(self.estimator.best_cv_score_)

    def predict(self, X):
        X_array = np.asarray(X, dtype=np.float32)
        return _to_prediction_vector(
            self.estimator.predict(X_array[:, self.feature_indices])
        )

    def selected_mask(self):
        subset_mask = _selected_mask_from_model(self.estimator)
        full_mask = np.zeros(self.n_features, dtype=bool)
        full_mask[self.feature_indices[subset_mask]] = True
        return full_mask

    def skip_importance(self):
        weights = (
            self.estimator.model.skip.weight.detach()
            .norm(p=2, dim=0)
            .cpu()
            .numpy()
        )
        full_importance = np.zeros(self.n_features, dtype=np.float64)
        full_importance[self.feature_indices] = weights
        return full_importance


class PTLassoOrientedTwoStageLassoNetRegressorCV(
    PTLassoOrientedTwoStageLassoNetRegressor
):
    """
    ptLasso-oriented two-stage estimator using official CV models in both stages.

    Lambda selection is delegated to ``LassoNetRegressorCV`` for the shared
    model and for every group correction model. Alpha is selected with an outer
    stratified K-fold loop using the unweighted mean validation MSE across folds.
    Thus Stage 2 still receives both forms of Stage 1 transfer information:

    ``offset = (1 - alpha) * f0(X)``

    and feature weights equal to 1 on the Stage 1 support and ``1 / alpha``
    outside it. At alpha=0, Stage 2 is fitted only on Stage 1-selected columns.
    """

    def __init__(
        self,
        *,
        alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        alpha_cv=3,
        stage1_cv=5,
        stage2_cv=5,
        stage1_M=10.0,
        stage2_M=10.0,
        common_model_kwargs=None,
        group_model_kwargs=None,
        random_state=123,
        verbose=1,
        precomputed_alpha_fold_mses=None,
    ):
        alpha_values = tuple(float(value) for value in alpha_grid)
        if not alpha_values:
            raise ValueError("alpha_grid must contain at least one value.")
        if any(value < 0.0 or value > 1.0 for value in alpha_values):
            raise ValueError("Every alpha value must be in [0, 1].")

        super().__init__(
            alpha=alpha_values[0],
            stage1_M=stage1_M,
            stage2_M=stage2_M,
            common_model_kwargs=common_model_kwargs,
            group_model_kwargs=group_model_kwargs,
        )
        self.alpha_grid = alpha_values
        self.alpha_cv = alpha_cv
        self.stage1_cv = stage1_cv
        self.stage2_cv = stage2_cv
        self.random_state = random_state
        self.verbose = int(verbose)
        self.precomputed_alpha_fold_mses = {
            int(fold): {
                float(alpha): float(mse)
                for alpha, mse in fold_values.items()
            }
            for fold, fold_values in (precomputed_alpha_fold_mses or {}).items()
        }

        self.best_alpha_ = None
        self.best_stage1_lambda_ = None
        self.best_stage2_lambda_by_group_ = {}
        self.alpha_cv_results_ = []
        self.alpha_cv_details_ = []
        self.fit_elapsed_seconds_ = None

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _new_stage1_model(self):
        kwargs = _translate_cv_kwargs(
            self.common_model_kwargs,
            default_M=self.stage1_M,
        )
        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("random_state", self.random_state)
        kwargs.setdefault("torch_seed", self.random_state)
        return LassoNetRegressorCV(cv=self.stage1_cv, **kwargs)

    def _new_stage2_model(self, penalty_factor):
        kwargs = _translate_cv_kwargs(
            self.group_model_kwargs,
            default_M=self.stage2_M,
        )
        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("random_state", self.random_state)
        kwargs.setdefault("torch_seed", self.random_state)
        penalty_factor = np.asarray(penalty_factor, dtype=np.float32)
        if _official_supports_penalty_factor():
            kwargs["penalty_factor"] = penalty_factor
        elif not np.allclose(penalty_factor, 1.0):
            raise RuntimeError(
                "Weighted Stage 2 penalties require the local LassoNet fork "
                "with penalty_factor support."
            )
        return LassoNetRegressorCV(cv=self.stage2_cv, **kwargs)

    def _fit_stage2_group(
        self,
        X,
        y,
        stage1_prediction,
        common_support,
        *,
        alpha,
    ):
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        stage1_prediction = np.asarray(stage1_prediction, dtype=np.float32).reshape(-1)
        residual = y_array - (1.0 - alpha) * stage1_prediction

        penalty_factor, _ = _build_penalty_factor(alpha, common_support)
        if alpha == 0.0:
            feature_indices = np.flatnonzero(common_support)
            if feature_indices.size == 0:
                raise RuntimeError(
                    "alpha=0 requires Stage 1 to select at least one feature."
                )
        else:
            feature_indices = np.arange(X_array.shape[1], dtype=int)

        subset_penalty = penalty_factor[feature_indices]
        model = self._new_stage2_model(subset_penalty)
        model.fit(X_array[:, feature_indices], residual)
        return _OfficialCVModelAdapter(model, feature_indices, X_array.shape[1])

    def _alpha_splitter(self, X, groups):
        if isinstance(self.alpha_cv, int):
            return StratifiedKFold(
                n_splits=self.alpha_cv,
                shuffle=True,
                random_state=self.random_state,
            ).split(X, groups)
        return check_cv(self.alpha_cv).split(X, groups)

    def _select_alpha(self, X, y, groups):
        fold_mses = {alpha: [] for alpha in self.alpha_grid}
        detail_rows = []

        for fold_index, (train_index, val_index) in enumerate(
            self._alpha_splitter(X, groups),
            start=1,
        ):
            fold_start = time.monotonic()
            precomputed = self.precomputed_alpha_fold_mses.get(fold_index, {})
            missing_alphas = []
            for alpha in self.alpha_grid:
                if alpha not in precomputed:
                    missing_alphas.append(alpha)
                    continue
                fold_mse = float(precomputed[alpha])
                fold_mses[alpha].append(fold_mse)
                detail_rows.append(
                    {
                        "fold": fold_index,
                        "alpha": alpha,
                        "validation_mse": fold_mse,
                        "stage1_best_lambda": np.nan,
                        "stage1_selected_count": np.nan,
                        "stage2_best_lambda_by_group": {},
                        "elapsed_seconds": 0.0,
                        "precomputed": True,
                    }
                )
                self._log(
                    f"[Alpha CV] fold={fold_index}, alpha={alpha:.2f}: "
                    f"reusing validation MSE={fold_mse:.6f}"
                )

            if not missing_alphas:
                self._log(
                    f"[Alpha CV] fold={fold_index}: all alpha scores were reused"
                )
                continue

            self._log(f"[Alpha CV] fold={fold_index}: fitting official Stage 1 CV")
            stage1_model = self._new_stage1_model()
            stage1_model.fit(X[train_index], y[train_index])
            common_support = _selected_mask_from_model(stage1_model)
            stage1_train_prediction = _to_prediction_vector(
                stage1_model.predict(X[train_index])
            )
            stage1_val_prediction = _to_prediction_vector(
                stage1_model.predict(X[val_index])
            )
            self._log(
                f"[Alpha CV] fold={fold_index}: Stage 1 lambda="
                f"{float(stage1_model.best_lambda_):.6g}, selected={int(common_support.sum())}"
            )

            train_groups = groups[train_index]
            val_groups = groups[val_index]
            for alpha in missing_alphas:
                alpha_start = time.monotonic()
                fold_prediction = np.full(len(val_index), np.nan, dtype=np.float32)
                group_lambdas = {}

                for group in np.unique(train_groups):
                    train_group_mask = train_groups == group
                    val_group_mask = val_groups == group
                    if not val_group_mask.any():
                        continue

                    self._log(
                        f"[Alpha CV] fold={fold_index}, alpha={alpha:.2f}, "
                        f"group={group}: fitting official Stage 2 CV"
                    )
                    group_model = self._fit_stage2_group(
                        X[train_index][train_group_mask],
                        y[train_index][train_group_mask],
                        stage1_train_prediction[train_group_mask],
                        common_support,
                        alpha=alpha,
                    )
                    group_lambdas[str(group)] = group_model.best_lambda_
                    correction = _to_prediction_vector(
                        group_model.predict(X[val_index][val_group_mask])
                    )
                    fold_prediction[val_group_mask] = (
                        (1.0 - alpha) * stage1_val_prediction[val_group_mask]
                        + correction
                    )

                if np.isnan(fold_prediction).any():
                    missing = np.unique(val_groups[np.isnan(fold_prediction)]).tolist()
                    raise RuntimeError(
                        f"No Stage 2 model was available for validation groups: {missing}."
                    )

                fold_error = (y[val_index] - fold_prediction) ** 2
                fold_mse = float(np.mean(fold_error))
                fold_mses[alpha].append(fold_mse)
                detail_rows.append(
                    {
                        "fold": fold_index,
                        "alpha": alpha,
                        "validation_mse": fold_mse,
                        "stage1_best_lambda": float(stage1_model.best_lambda_),
                        "stage1_selected_count": int(common_support.sum()),
                        "stage2_best_lambda_by_group": group_lambdas,
                        "elapsed_seconds": time.monotonic() - alpha_start,
                        "precomputed": False,
                    }
                )
                self._log(
                    f"[Alpha CV] fold={fold_index}, alpha={alpha:.2f}: "
                    f"validation MSE={fold_mse:.6f}, elapsed="
                    f"{time.monotonic() - alpha_start:.1f}s"
                )

            self._log(
                f"[Alpha CV] fold={fold_index}: complete in "
                f"{time.monotonic() - fold_start:.1f}s"
            )

        results = []
        for alpha in self.alpha_grid:
            alpha_fold_mses = np.asarray(fold_mses[alpha], dtype=np.float64)
            if alpha_fold_mses.size == 0 or not np.isfinite(alpha_fold_mses).all():
                raise RuntimeError(f"Incomplete validation MSE values for alpha={alpha}.")
            mean_fold_mse = float(np.mean(alpha_fold_mses))
            fold_mse_std = (
                float(np.std(alpha_fold_mses, ddof=1))
                if alpha_fold_mses.size > 1
                else 0.0
            )
            results.append(
                {
                    "alpha": alpha,
                    "mean_fold_mse": mean_fold_mse,
                    "root_mean_fold_mse": float(np.sqrt(mean_fold_mse)),
                    "fold_mse_std": fold_mse_std,
                    "fold_mse_se": float(
                        fold_mse_std / np.sqrt(alpha_fold_mses.size)
                    ),
                    "n_folds": int(alpha_fold_mses.size),
                }
            )

        self.alpha_cv_details_ = detail_rows
        self.alpha_cv_results_ = results
        return min(results, key=lambda row: row["mean_fold_mse"])["alpha"]

    def fit(self, X, y, groups, *, feature_names=None):
        start = time.monotonic()
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        groups_array = _to_numpy_groups(groups)

        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if len(X_array) != len(y_array) or len(X_array) != len(groups_array):
            raise ValueError("X, y, and groups must have the same number of rows.")
        if not _official_supports_penalty_factor() and any(alpha < 1.0 for alpha in self.alpha_grid):
            raise RuntimeError(
                "Stage 2 support weighting requires the local LassoNet fork with "
                "penalty_factor support. The upstream package does not expose it."
            )

        self.feature_names_ = _resolve_feature_names(feature_names, X_array.shape[1])
        self._log(
            f"[Two-stage CV] alpha grid={list(self.alpha_grid)}; "
            f"alpha_cv={self.alpha_cv}; stage1_cv={self.stage1_cv}; stage2_cv={self.stage2_cv}"
        )
        self.best_alpha_ = float(self._select_alpha(X_array, y_array, groups_array))
        self.alpha = self.best_alpha_
        best_alpha_row = next(
            row for row in self.alpha_cv_results_ if row["alpha"] == self.best_alpha_
        )
        self._log(
            f"[Two-stage CV] selected alpha={self.best_alpha_:.2f} with mean "
            f"fold MSE={best_alpha_row['mean_fold_mse']:.6f}"
        )

        self._log("[Final fit] fitting official Stage 1 CV on all development data")
        self.common_model_ = self._new_stage1_model()
        self.common_model_.fit(X_array, y_array)
        self.best_stage1_lambda_ = float(self.common_model_.best_lambda_)
        self.stage1_artifacts_ = extract_stage1_artifacts(
            self.common_model_,
            X_array,
            alpha=self.best_alpha_,
        )
        self.common_support_ = self.stage1_artifacts_["common_support"]
        self.penalty_factor_ = self.stage1_artifacts_["penalty_factor"]

        self.group_order_ = np.unique(groups_array)
        self.group_models_ = {}
        self.best_stage2_lambda_by_group_ = {}
        stage1_prediction = self.stage1_artifacts_["common_predictions"]
        for group in self.group_order_:
            group_mask = groups_array == group
            self._log(f"[Final fit] group={group}: fitting official Stage 2 CV")
            group_model = self._fit_stage2_group(
                X_array[group_mask],
                y_array[group_mask],
                stage1_prediction[group_mask],
                self.common_support_,
                alpha=self.best_alpha_,
            )
            self.group_models_[group] = group_model
            self.best_stage2_lambda_by_group_[group] = group_model.best_lambda_
            self._log(
                f"[Final fit] group={group}: lambda={group_model.best_lambda_:.6g}, "
                f"selected={int(group_model.selected_mask().sum())}"
            )

        self.fit_elapsed_seconds_ = time.monotonic() - start
        self._log(
            f"[Two-stage CV] complete in {self.fit_elapsed_seconds_:.1f}s; "
            f"Stage 1 lambda={self.best_stage1_lambda_:.6g}, "
            f"common selected={int(self.common_support_.sum())}"
        )
        return self
