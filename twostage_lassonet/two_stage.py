from copy import deepcopy
from functools import partial
import os
from pathlib import Path

import numpy as np
import torch

from .regression import LassoNetRegressor as Stage2LassoNetRegressor

_mplconfig_dir = Path.cwd() / ".mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

try:
    from lassonet import LassoNetRegressor as Stage1LassoNetRegressor
except ImportError as exc:  # pragma: no cover - exercised when dependency is missing
    raise ImportError(
        "twostage_lassonet.two_stage requires the official 'lassonet' package "
        "for stage-1 training."
    ) from exc


def _to_numpy_groups(groups):
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError("groups must be a 1D array-like object.")
    return group_array


def _to_prediction_vector(predictions):
    prediction_array = np.asarray(predictions, dtype=np.float32)
    if prediction_array.ndim == 2 and prediction_array.shape[1] == 1:
        prediction_array = prediction_array.reshape(-1)
    if prediction_array.ndim != 1:
        raise ValueError("Expected stage-1 predictions to have shape (n_samples,) or (n_samples, 1).")
    return prediction_array


def _build_penalty_factor(alpha, common_support):
    if alpha == 0.0:
        penalty_factor = np.ones(common_support.shape[0], dtype=np.float32)
        zero_mask = ~common_support
    else:
        penalty_factor = np.full(
            common_support.shape[0],
            1.0 / alpha,
            dtype=np.float32,
        )
        penalty_factor[common_support] = 1.0
        zero_mask = np.zeros(common_support.shape[0], dtype=bool)
    return penalty_factor, zero_mask


def _selected_mask_from_stage1_model(stage1_model):
    return _selected_mask_from_model(stage1_model)


def _selected_mask_from_model(model):
    """Read a support mask from either our estimator or official LassoNet CV."""
    if hasattr(model, "selected_mask"):
        selected = model.selected_mask()
    elif getattr(model, "best_selected_", None) is not None:
        selected = model.best_selected_
    elif getattr(model, "model", None) is not None:
        selected = model.model.input_mask()
    else:
        raise ValueError("The LassoNet model must be fitted before extracting support information.")

    if torch.is_tensor(selected):
        selected = selected.detach().cpu().numpy()
    selected = np.asarray(selected, dtype=bool)
    if selected.ndim != 1:
        raise ValueError("Expected a one-dimensional feature support mask.")
    return selected


def _build_optimizers(learning_rate, path_learning_rate, momentum):
    return (
        partial(torch.optim.Adam, lr=learning_rate),
        partial(torch.optim.SGD, lr=path_learning_rate, momentum=momentum),
    )


def _translate_stage1_kwargs(common_model_kwargs, *, stage1_lambda, stage1_M):
    kwargs = deepcopy(common_model_kwargs or {})

    translated = {
        "hidden_dims": kwargs.pop("hidden_dims", (100,)),
        "M": kwargs.pop("M", stage1_M),
        "dropout": kwargs.pop("dropout", 0) or 0,
    }

    if "n_iters" in kwargs:
        translated["n_iters"] = kwargs.pop("n_iters")
    else:
        dense_epochs = kwargs.pop("dense_epochs", None)
        sparse_epochs = kwargs.pop("sparse_epochs", None)
        if dense_epochs is not None or sparse_epochs is not None:
            translated["n_iters"] = (
                1000 if dense_epochs is None else dense_epochs,
                100 if sparse_epochs is None else sparse_epochs,
            )

    if "patience" in kwargs:
        translated["patience"] = kwargs.pop("patience")
    else:
        dense_patience = kwargs.pop("dense_patience", None)
        sparse_patience = kwargs.pop("sparse_patience", None)
        if dense_patience is not None or sparse_patience is not None:
            translated["patience"] = (
                dense_patience,
                sparse_patience,
            )

    if "optim" in kwargs:
        translated["optim"] = kwargs.pop("optim")
    else:
        learning_rate = kwargs.pop("learning_rate", None)
        path_learning_rate = kwargs.pop("path_learning_rate", None)
        momentum = kwargs.pop("momentum", 0.9)
        if learning_rate is not None or path_learning_rate is not None:
            init_lr = 1e-3 if learning_rate is None else learning_rate
            sparse_lr = init_lr if path_learning_rate is None else path_learning_rate
            translated["optim"] = _build_optimizers(init_lr, sparse_lr, momentum)

    passthrough_keys = (
        "lambda_start",
        "lambda_seq",
        "gamma",
        "gamma_skip",
        "path_multiplier",
        "penalty_factor",
        "groups",
        "batch_size",
        "backtrack",
        "val_size",
        "device",
        "verbose",
        "random_state",
        "torch_seed",
    )
    for key in passthrough_keys:
        if key in kwargs:
            translated[key] = kwargs.pop(key)

    if "lambda_seq" not in translated and "lambda_start" not in translated:
        translated["lambda_seq"] = [float(stage1_lambda)]

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported stage-1 kwargs for official lassonet: {unknown}.")

    return translated


def extract_stage1_artifacts(stage1_model, X, *, alpha, X_val=None):
    X_array = np.asarray(X, dtype=np.float32)
    common_support = _selected_mask_from_stage1_model(stage1_model)
    penalty_factor, zero_mask = _build_penalty_factor(alpha, common_support)

    common_predictions = _to_prediction_vector(stage1_model.predict(X_array))
    offset = (1.0 - alpha) * common_predictions

    artifacts = {
        "common_support": common_support,
        "penalty_factor": penalty_factor,
        "zero_mask": zero_mask,
        "common_predictions": common_predictions,
        "offset": offset.astype(np.float32, copy=False),
    }

    if X_val is not None:
        X_val_array = np.asarray(X_val, dtype=np.float32)
        common_predictions_val = _to_prediction_vector(stage1_model.predict(X_val_array))
        offset_val = (1.0 - alpha) * common_predictions_val
        artifacts["common_predictions_val"] = common_predictions_val
        artifacts["offset_val"] = offset_val.astype(np.float32, copy=False)

    return artifacts


def fit_stage2_group_models(
    X,
    y,
    groups,
    *,
    offset,
    penalty_factor,
    zero_mask,
    stage2_lambda=1e-2,
    stage2_M=10.0,
    group_model_kwargs=None,
    X_val=None,
    y_val=None,
    groups_val=None,
    offset_val=None,
):
    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    groups_array = _to_numpy_groups(groups)

    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X_array.shape[0] != groups_array.shape[0]:
        raise ValueError("X and groups must have the same number of rows.")

    offset_array = np.asarray(offset, dtype=np.float32).reshape(-1)
    if offset_array.shape[0] != X_array.shape[0]:
        raise ValueError("offset must have one entry per training row.")

    if X_val is not None and y_val is not None:
        X_val_array = np.asarray(X_val, dtype=np.float32)
        y_val_array = np.asarray(y_val, dtype=np.float32)
        if groups_val is None:
            raise ValueError("groups_val must be supplied when X_val and y_val are supplied.")
        groups_val_array = _to_numpy_groups(groups_val)
        if offset_val is None:
            raise ValueError("offset_val must be supplied when using validation data.")
        offset_val_array = np.asarray(offset_val, dtype=np.float32).reshape(-1)
    else:
        X_val_array = y_val_array = groups_val_array = offset_val_array = None

    if isinstance(penalty_factor, np.ndarray):
        penalty_factor_array = penalty_factor.astype(np.float32, copy=False)
    else:
        penalty_factor_array = np.asarray(penalty_factor, dtype=np.float32)

    if isinstance(zero_mask, np.ndarray):
        zero_mask_array = zero_mask.astype(bool, copy=False)
    else:
        zero_mask_array = np.asarray(zero_mask, dtype=bool)

    def _resolve_group_value(value, group):
        if isinstance(value, dict):
            if group not in value:
                raise KeyError(f"Missing group-specific value for group {group!r}.")
            return value[group]
        return value

    group_order = np.unique(groups_array)
    group_models = {}

    for group in group_order:
        train_mask = groups_array == group
        group_y = y_array[train_mask] - offset_array[train_mask]

        X_group_val = None
        y_group_val = None
        if X_val_array is not None:
            val_mask = groups_val_array == group
            if val_mask.any():
                X_group_val = X_val_array[val_mask]
                y_group_val = y_val_array[val_mask] - offset_val_array[val_mask]

        group_model = Stage2LassoNetRegressor(**deepcopy(group_model_kwargs or {}))
        group_model.fit(
            X_array[train_mask],
            group_y,
            lambda_=_resolve_group_value(stage2_lambda, group),
            penalty_factor=penalty_factor_array,
            M=_resolve_group_value(stage2_M, group),
            zero_mask=zero_mask_array,
            X_val=X_group_val,
            y_val=y_group_val,
        )
        group_models[group] = group_model

    return {
        "group_models": group_models,
        "group_order": group_order,
    }


class TwoStagePretrainedLassoNetRegressor:
    """
    Two-stage pretrained LassoNet for grouped samples.

    Stage 1:
        fit a common model using the official `lassonet` package.

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
        self.stage1_artifacts_ = None

    def _build_penalty_factor(self, common_support):
        return _build_penalty_factor(self.alpha, common_support)

    def _fit_stage1_model(self, X, y, *, X_val=None, y_val=None):
        model = Stage1LassoNetRegressor(
            **_translate_stage1_kwargs(
                self.common_model_kwargs,
                stage1_lambda=self.stage1_lambda,
                stage1_M=self.stage1_M,
            )
        )
        model.fit(X, y, X_val=X_val, y_val=y_val)
        return model

    def fit(
        self,
        X,
        y,
        groups,
        *,
        X_val=None,
        y_val=None,
        groups_val=None,
        stage1_model=None,
        stage1_state=None,
    ):
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        groups_array = _to_numpy_groups(groups)

        if X_array.shape[0] != groups_array.shape[0]:
            raise ValueError("X and groups must have the same number of rows.")
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if stage1_model is None:
            self.common_model_ = self._fit_stage1_model(
                X_array,
                y_array,
                X_val=X_val,
                y_val=y_val,
            )
        else:
            self.common_model_ = stage1_model
            if stage1_state is not None:
                if not hasattr(self.common_model_, "load"):
                    raise TypeError("stage1_model does not support `load(state)`.")
                self.common_model_.load(stage1_state)

        self.stage1_artifacts_ = extract_stage1_artifacts(
            self.common_model_,
            X_array,
            alpha=self.alpha,
            X_val=X_val,
        )

        self.common_support_ = self.stage1_artifacts_["common_support"]
        self.penalty_factor_ = self.stage1_artifacts_["penalty_factor"]

        stage2_result = fit_stage2_group_models(
            X_array,
            y_array,
            groups_array,
            offset=self.stage1_artifacts_["offset"],
            penalty_factor=self.stage1_artifacts_["penalty_factor"],
            zero_mask=self.stage1_artifacts_["zero_mask"],
            stage2_lambda=self.stage2_lambda,
            stage2_M=self.stage2_M,
            group_model_kwargs=self.group_model_kwargs,
            X_val=X_val,
            y_val=y_val,
            groups_val=groups_val,
            offset_val=self.stage1_artifacts_.get("offset_val"),
        )

        self.group_models_ = stage2_result["group_models"]
        self.group_order_ = stage2_result["group_order"]

        return self

    def predict_components(self, X, groups):
        if self.common_model_ is None:
            raise RuntimeError("fit must be called before predict_components.")

        X_array = np.asarray(X, dtype=np.float32)
        groups_array = _to_numpy_groups(groups)
        if X_array.shape[0] != groups_array.shape[0]:
            raise ValueError("X and groups must have the same number of rows.")

        common_component = (1.0 - self.alpha) * _to_prediction_vector(
            self.common_model_.predict(X_array)
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
        return np.flatnonzero(_selected_mask_from_model(self.group_models_[group]))

    def get_group_individual_support(self, group):
        group_support = _selected_mask_from_model(self.group_models_[group])
        return np.flatnonzero(group_support & ~self.common_support_)

    def get_group_final_support(self, group):
        group_support = _selected_mask_from_model(self.group_models_[group])
        return np.flatnonzero(group_support | self.common_support_)
