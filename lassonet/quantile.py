from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .interfaces import BaseLassoNetCV, LassoNetRegressor


def _validate_tau(tau: float) -> float:
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1")
    return tau


def _coerce_tensor_shapes(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape == target.shape:
        return prediction, target
    if prediction.ndim == 1 and target.ndim == 2 and target.shape[1] == 1:
        prediction = prediction.view(-1, 1)
    elif target.ndim == 1 and prediction.ndim == 2 and prediction.shape[1] == 1:
        target = target.view(-1, 1)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} "
            f"does not match target shape {tuple(target.shape)}"
        )
    return prediction, target


def _coerce_array_shapes(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match "
            f"target shape {target.shape}"
        )
    return prediction, target


def _as_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def _pinball_values(residual: Any, tau: float, *, maximum):
    return maximum(tau * residual, (tau - 1.0) * residual)


class PinballLoss(nn.Module):
    """Mean quantile check loss for a fixed tau in (0, 1)."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = _validate_tau(tau)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction, target = _coerce_tensor_shapes(prediction, target)
        residual = target - prediction
        return _pinball_values(residual, self.tau, maximum=torch.maximum).mean()


class QuantileLassoNetRegressor(LassoNetRegressor):
    """LassoNet regressor trained with quantile check loss."""

    criterion = PinballLoss(0.5)

    def __init__(
        self,
        *,
        tau: float = 0.5,
        hidden_dims=(100,),
        lambda_start="auto",
        lambda_seq=None,
        gamma=0.0,
        gamma_skip=0.0,
        path_multiplier=1.02,
        M=10,
        penalty_factor=None,
        groups=None,
        dropout=0,
        batch_size=None,
        optim=None,
        n_iters=(1000, 100),
        patience=(100, 10),
        tol=0.99,
        backtrack=False,
        val_size=None,
        device=None,
        verbose=1,
        random_state=None,
        torch_seed=None,
    ) -> None:
        super().__init__(
            hidden_dims=hidden_dims,
            lambda_start=lambda_start,
            lambda_seq=lambda_seq,
            gamma=gamma,
            gamma_skip=gamma_skip,
            path_multiplier=path_multiplier,
            M=M,
            penalty_factor=penalty_factor,
            groups=groups,
            dropout=dropout,
            batch_size=batch_size,
            optim=optim,
            n_iters=n_iters,
            patience=patience,
            tol=tol,
            backtrack=backtrack,
            val_size=val_size,
            device=device,
            verbose=verbose,
            random_state=random_state,
            torch_seed=torch_seed,
        )
        self.tau = _validate_tau(tau)
        self.criterion = PinballLoss(self.tau)

    def score(self, X, y) -> float:
        """Negative pinball loss; larger is better for LassoNet CV."""
        prediction = _as_numpy_array(self.predict(X))
        target = _as_numpy_array(y)
        prediction, target = _coerce_array_shapes(prediction, target)
        residual = target - prediction
        loss = _pinball_values(residual, self.tau, maximum=np.maximum)
        return -float(loss.mean())


class QuantileLassoNetRegressorCV(
    BaseLassoNetCV,
    QuantileLassoNetRegressor,
):
    """Internal LassoNet cross-validation using pinball loss."""

    def __init__(
        self,
        cv=None,
        *,
        tau: float = 0.5,
        hidden_dims=(100,),
        lambda_start="auto",
        lambda_seq=None,
        gamma=0.0,
        gamma_skip=0.0,
        path_multiplier=1.02,
        M=10,
        penalty_factor=None,
        groups=None,
        dropout=0,
        batch_size=None,
        optim=None,
        n_iters=(1000, 100),
        patience=(100, 10),
        tol=0.99,
        backtrack=False,
        val_size=None,
        device=None,
        verbose=1,
        random_state=None,
        torch_seed=None,
    ) -> None:
        super().__init__(
            cv=cv,
            tau=tau,
            hidden_dims=hidden_dims,
            lambda_start=lambda_start,
            lambda_seq=lambda_seq,
            gamma=gamma,
            gamma_skip=gamma_skip,
            path_multiplier=path_multiplier,
            M=M,
            penalty_factor=penalty_factor,
            groups=groups,
            dropout=dropout,
            batch_size=batch_size,
            optim=optim,
            n_iters=n_iters,
            patience=patience,
            tol=tol,
            backtrack=backtrack,
            val_size=val_size,
            device=device,
            verbose=verbose,
            random_state=random_state,
            torch_seed=torch_seed,
        )
