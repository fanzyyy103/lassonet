import numpy as np
import torch
from sklearn.datasets import load_diabetes, load_digits

from lassonet import LassoNetClassifier, LassoNetRegressor
from lassonet.prox import prox


def test_regressor():
    X, y = load_diabetes(return_X_y=True)
    model = LassoNetRegressor()
    model.fit(X, y)
    model.score(X, y)


def test_classifier():
    X, y = load_digits(return_X_y=True)
    model = LassoNetClassifier()
    model.fit(X, y)
    model.score(X, y)


def test_offset_matches_residual_fit():
    X, y = load_diabetes(return_X_y=True)
    offset = 0.25 * y

    model_with_offset = LassoNetRegressor(
        hidden_dims=(10,),
        lambda_seq=[1e-2],
        n_iters=(5, 5),
        patience=(2, 2),
        random_state=0,
        torch_seed=0,
        verbose=0,
    )
    path_with_offset = model_with_offset.path(X, y, offset=offset)
    model_with_offset.load(path_with_offset[-1].state_dict)

    model_on_residual = LassoNetRegressor(
        hidden_dims=(10,),
        lambda_seq=[1e-2],
        n_iters=(5, 5),
        patience=(2, 2),
        random_state=0,
        torch_seed=0,
        verbose=0,
    )
    path_on_residual = model_on_residual.path(X, y - offset)
    model_on_residual.load(path_on_residual[-1].state_dict)

    pred_with_offset = model_with_offset.predict(X)
    pred_on_residual = model_on_residual.predict(X)
    np.testing.assert_allclose(pred_with_offset, pred_on_residual, atol=1e-5)


def test_weighted_prox_uses_feature_specific_thresholds():
    v = torch.tensor([[3.0, 3.0]])
    u = torch.tensor([[3.0, 3.0]])
    unweighted_beta, unweighted_theta = prox(v, u, lambda_=1.0, lambda_bar=0.0, M=1.0)
    weighted_beta, weighted_theta = prox(
        v,
        u,
        lambda_=torch.tensor([1.0, 2.0]),
        lambda_bar=0.0,
        M=1.0,
    )

    torch.testing.assert_close(weighted_beta[:, 0], unweighted_beta[:, 0])
    torch.testing.assert_close(weighted_theta[:, 0], unweighted_theta[:, 0])
    assert torch.abs(weighted_beta[:, 1]).item() < torch.abs(unweighted_beta[:, 1]).item()
    assert torch.abs(weighted_theta[:, 1]).item() < torch.abs(unweighted_theta[:, 1]).item()
