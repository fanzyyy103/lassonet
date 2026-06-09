import torch
from torch.nn import functional as F


def soft_threshold(threshold, value):
    return torch.sign(value) * torch.relu(torch.abs(value) - threshold)


def sign_binary(value):
    ones = torch.ones_like(value)
    return torch.where(value >= 0, ones, -ones)


def _coerce_parameter(value, like, batch):
    if torch.is_tensor(value):
        tensor = value.to(device=like.device, dtype=like.dtype)
    else:
        tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)

    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    elif tensor.ndim == 1:
        if tensor.shape[0] != batch:
            raise ValueError(
                f"Expected parameter with {batch} entries, got {tensor.shape[0]}."
            )
    else:
        raise ValueError("Parameters for prox must be scalars or 1D tensors.")

    return tensor.view(1, batch)


def prox(v, u, *, lambda_, lambda_bar=0.0, M=1.0):
    """
    Column-wise proximal operator for the LassoNet hierarchy constraint.

    Parameters
    ----------
    v : torch.Tensor
        Skip-layer weights with shape (n_outputs, n_features) or (n_outputs,).
    u : torch.Tensor
        First hidden layer weights with shape (n_hidden, n_features) or (n_hidden,).
    lambda_ : float or torch.Tensor
        L1 penalty applied to the skip-layer columns. Can be feature-specific.
    lambda_bar : float or torch.Tensor, default=0.0
        Additional soft-thresholding term on the first hidden layer.
    M : float or torch.Tensor, default=1.0
        Hierarchy parameter. Can be feature-specific.
    """
    one_dimensional = v.ndim == 1
    if one_dimensional:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    _, batch = u.shape
    lambda_ = _coerce_parameter(lambda_, v, batch)
    lambda_bar = _coerce_parameter(lambda_bar, v, batch)
    M = _coerce_parameter(M, v, batch)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    hidden_size = u.shape[0]
    s = torch.arange(hidden_size + 1.0, device=v.device, dtype=v.dtype).view(-1, 1)
    zeros = torch.zeros(1, batch, device=v.device, dtype=v.dtype)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)],
        dim=0,
    )

    norm_v = torch.norm(v, p=2, dim=0, keepdim=True).clamp_min(1e-12)

    x = F.relu(1 - a_s / norm_v) / (1 + s * M.square())
    w = M * x * norm_v

    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros], dim=0)

    idx = torch.sum(lower > w, dim=0, keepdim=True)

    x_star = torch.gather(x, 0, idx)
    w_star = torch.gather(w, 0, idx)

    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(
        soft_threshold(lambda_bar, u.abs()),
        w_star,
    )

    if one_dimensional:
        beta_star = beta_star.squeeze(-1)
        theta_star = theta_star.squeeze(-1)

    return beta_star, theta_star


def inplace_prox(skip_layer, first_hidden_layer, lambda_, lambda_bar=0.0, M=1.0):
    skip_layer.weight.data, first_hidden_layer.weight.data = prox(
        skip_layer.weight.data,
        first_hidden_layer.weight.data,
        lambda_=lambda_,
        lambda_bar=lambda_bar,
        M=M,
    )
