from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .model import LassoNet


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    objective: float
    loss: float
    val_objective: float
    val_loss: float
    regularization: float
    selected: torch.BoolTensor
    n_iters: int


def _as_float_tensor(array, device):
    if torch.is_tensor(array):
        return array.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array, device=device, dtype=torch.float32)


def _split_train_val(X, y, val_size, seed):
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1) when no explicit validation set is supplied.")

    n_samples = X.shape[0]
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_samples)
    n_val = max(1, int(round(n_samples * val_size)))
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    if train_idx.size == 0:
        train_idx = val_idx
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def _zero_forbidden_features(model, zero_mask):
    if zero_mask is None:
        return

    model.skip.weight.data[:, zero_mask] = 0.0
    model.layers[0].weight.data[:, zero_mask] = 0.0


class LassoNetRegressor:
    def __init__(
        self,
        *,
        hidden_dims=(100,),
        dropout=None,
        M=10.0,
        batch_size=None,
        learning_rate=1e-3,
        path_learning_rate=1e-3,
        momentum=0.9,
        dense_epochs=500,
        sparse_epochs=200,
        dense_patience=50,
        sparse_patience=20,
        tol=0.99,
        val_size=0.1,
        backtrack=True,
        device=None,
        verbose=0,
        torch_seed=None,
        random_state=None,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.M = M
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.path_learning_rate = path_learning_rate
        self.momentum = momentum
        self.dense_epochs = dense_epochs
        self.sparse_epochs = sparse_epochs
        self.dense_patience = dense_patience
        self.sparse_patience = sparse_patience
        self.tol = tol
        self.val_size = val_size
        self.backtrack = backtrack
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.verbose = verbose
        self.torch_seed = torch_seed
        self.random_state = random_state

        self.model = None
        self.output_dim_ = None
        self.squeeze_output_ = None
        self.history_ = []
        self.lambda_ = None
        self.penalty_factor_ = None

    def _prepare_y(self, y):
        y_array = np.asarray(y, dtype=np.float32)
        self.squeeze_output_ = y_array.ndim == 1
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        return y_array

    def _init_model(self, input_dim, output_dim):
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
        self.model = LassoNet(
            input_dim,
            *self.hidden_dims,
            output_dim,
            dropout=self.dropout,
        ).to(self.device)

    def _cast_input(self, X, y=None):
        X_tensor = _as_float_tensor(np.asarray(X, dtype=np.float32), self.device)
        if y is None:
            return X_tensor
        y_tensor = _as_float_tensor(np.asarray(y, dtype=np.float32), self.device)
        return X_tensor, y_tensor

    def _normalize_feature_vector(self, value, n_features, *, name):
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.repeat(n_features)
        elif tensor.ndim == 1 and tensor.numel() == n_features:
            tensor = tensor.clone()
        else:
            raise ValueError(f"{name} must be a scalar or a vector with length n_features.")
        return tensor

    def _validation_objective(self, X_val, y_val, lambda_, penalty_factor):
        with torch.no_grad():
            predictions = self.model(X_val)
            data_loss = torch.nn.functional.mse_loss(predictions, y_val, reduction="mean")
            regularization = self.model.skip_regularization(penalty_factor)
            return data_loss + lambda_ * regularization

    def _run_training_loop(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        lambda_,
        penalty_factor,
        M_vector,
        zero_mask,
        epochs,
        patience,
        optimizer,
    ):
        best_val = self._validation_objective(X_val, y_val, lambda_, penalty_factor).item()
        best_state = self.model.cpu_state_dict()
        epochs_since_improvement = 0
        best_loss = float("inf")
        best_iters = 0

        n_train = X_train.shape[0]
        batch_size = n_train if self.batch_size is None else min(self.batch_size, n_train)

        for epoch in range(epochs):
            indices = torch.randperm(n_train, device=self.device)
            running_loss = 0.0
            self.model.train()

            for start in range(0, n_train, batch_size):
                batch = indices[start : start + batch_size]
                optimizer.zero_grad()
                predictions = self.model(X_train[batch])
                data_loss = torch.nn.functional.mse_loss(
                    predictions,
                    y_train[batch],
                    reduction="mean",
                )
                data_loss.backward()
                optimizer.step()

                _zero_forbidden_features(self.model, zero_mask)
                if lambda_ > 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    self.model.prox(
                        lambda_=lambda_ * current_lr * penalty_factor,
                        M=M_vector,
                    )
                    _zero_forbidden_features(self.model, zero_mask)

                running_loss += data_loss.item() * batch.numel() / n_train

            self.model.eval()
            val_objective = self._validation_objective(X_val, y_val, lambda_, penalty_factor).item()

            if val_objective < self.tol * best_val:
                best_val = val_objective
                best_state = self.model.cpu_state_dict()
                best_loss = running_loss
                best_iters = epoch + 1
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if patience is not None and epochs_since_improvement >= patience:
                break

        if self.backtrack:
            self.model.load_state_dict(best_state)
        else:
            best_state = self.model.cpu_state_dict()
            best_loss = running_loss
            best_iters = epoch + 1
            best_val = self._validation_objective(X_val, y_val, lambda_, penalty_factor).item()

        with torch.no_grad():
            regularization = self.model.skip_regularization(penalty_factor).item()
            val_loss = best_val - lambda_ * regularization
            return HistoryItem(
                lambda_=lambda_,
                state_dict=best_state,
                objective=best_loss + lambda_ * regularization,
                loss=best_loss,
                val_objective=best_val,
                val_loss=val_loss,
                regularization=regularization,
                selected=self.model.input_mask().cpu(),
                n_iters=best_iters,
            )

    def fit(
        self,
        X,
        y,
        *,
        lambda_,
        penalty_factor: Optional[Sequence[float]] = None,
        M=None,
        zero_mask: Optional[Sequence[bool]] = None,
        X_val=None,
        y_val=None,
    ):
        X_array = np.asarray(X, dtype=np.float32)
        y_array = self._prepare_y(y)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if X_val is None or y_val is None:
            if self.val_size == 0:
                X_train, X_valid, y_train, y_valid = X_array, X_array, y_array, y_array
            else:
                X_train, X_valid, y_train, y_valid = _split_train_val(
                    X_array,
                    y_array,
                    self.val_size,
                    self.random_state,
                )
        else:
            X_train, y_train = X_array, y_array
            X_valid = np.asarray(X_val, dtype=np.float32)
            y_valid = self._prepare_y(y_val)

        self.output_dim_ = y_train.shape[1]
        self._init_model(X_train.shape[1], self.output_dim_)

        penalty_factor_tensor = self._normalize_feature_vector(
            1.0 if penalty_factor is None else penalty_factor,
            X_train.shape[1],
            name="penalty_factor",
        )
        M_tensor = self._normalize_feature_vector(
            self.M if M is None else M,
            X_train.shape[1],
            name="M",
        )
        zero_mask_tensor = None
        if zero_mask is not None:
            zero_mask_array = np.asarray(zero_mask, dtype=bool)
            if zero_mask_array.shape != (X_train.shape[1],):
                raise ValueError("zero_mask must have shape (n_features,).")
            zero_mask_tensor = torch.as_tensor(
                zero_mask_array,
                device=self.device,
                dtype=torch.bool,
            )

        X_train_tensor, y_train_tensor = self._cast_input(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self._cast_input(X_valid, y_valid)

        self.history_ = []
        if self.dense_epochs > 0:
            dense_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.history_.append(
                self._run_training_loop(
                    X_train_tensor,
                    y_train_tensor,
                    X_valid_tensor,
                    y_valid_tensor,
                    lambda_=0.0,
                    penalty_factor=penalty_factor_tensor,
                    M_vector=M_tensor,
                    zero_mask=zero_mask_tensor,
                    epochs=self.dense_epochs,
                    patience=self.dense_patience,
                    optimizer=dense_optimizer,
                )
            )

        sparse_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.path_learning_rate,
            momentum=self.momentum,
        )
        self.history_.append(
            self._run_training_loop(
                X_train_tensor,
                y_train_tensor,
                X_valid_tensor,
                y_valid_tensor,
                lambda_=float(lambda_),
                penalty_factor=penalty_factor_tensor,
                M_vector=M_tensor,
                zero_mask=zero_mask_tensor,
                epochs=self.sparse_epochs,
                patience=self.sparse_patience,
                optimizer=sparse_optimizer,
            )
        )

        self.lambda_ = float(lambda_)
        self.penalty_factor_ = penalty_factor_tensor.detach().cpu()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self._cast_input(X)).cpu().numpy()
        if self.squeeze_output_:
            return predictions.reshape(-1)
        return predictions

    def selected_mask(self):
        return self.model.input_mask().detach().cpu().numpy()

    def selected_indices(self):
        return np.flatnonzero(self.selected_mask())

    def load(self, state_dict):
        if self.model is None:
            output_dim, input_dim = state_dict["skip.weight"].shape
            self._init_model(input_dim, output_dim)
        self.model.load_state_dict(state_dict)
        return self
