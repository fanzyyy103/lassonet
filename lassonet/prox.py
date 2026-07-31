import torch
from torch.nn import functional as F


def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)


def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def _as_batch_vector(value, *, batch, ref, name):
    """Broadcast a scalar or per-feature vector to the prox batch dimension."""
    if torch.is_tensor(value):
        vector = value.to(device=ref.device, dtype=ref.dtype)
    else:
        vector = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)

    if vector.ndim == 0:
        vector = vector.repeat(batch)
    elif vector.ndim != 1 or vector.numel() != batch:
        raise ValueError(
            f"{name} must be a scalar or a 1D tensor with length {batch}."
        )

    return vector.view(1, batch)


def prox(v, u, *, lambda_, lambda_bar, M):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors

    This supports the weighted LassoNet / pretrained LassoNet extension by
    allowing lambda_ to be feature-specific. The only mathematical change is
    replacing a single threshold lambda with per-feature thresholds
    lambda_j = lambda * penalty_factor[j] inside the hierarchical prox step.
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape
    lambda_ = _as_batch_vector(lambda_, batch=batch, ref=u, name="lambda_")
    lambda_bar = _as_batch_vector(lambda_bar, batch=batch, ref=u, name="lambda_bar")
    M = _as_batch_vector(M, batch=batch, ref=u, name="M")

    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, batch).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)

    x = F.relu(1 - a_s / norm_v) / (1 + s * M**2)

    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star


def inplace_prox(beta, theta, lambda_, lambda_bar, M):
    beta.weight.data, theta.weight.data = prox(
        beta.weight.data, theta.weight.data, lambda_=lambda_, lambda_bar=lambda_bar, M=M
    )


def inplace_group_prox(groups, beta, theta, lambda_, lambda_bar, M):
    """
    groups is an iterable such that group[i] contains the indices of features in group i
    """

    def select_group_value(value, group_indices, *, name):
        if not torch.is_tensor(value):
            return value
        if value.ndim == 0:
            return value
        group_value = value[group_indices]
        if group_value.numel() == 0:
            raise ValueError(f"{name} cannot be empty for a feature group.")
        if not torch.allclose(group_value, group_value[0].expand_as(group_value)):
            raise ValueError(
                f"Weighted grouped LassoNet requires {name} to be constant within each feature group."
            )
        return group_value[0]

    beta_ = beta.weight.data
    theta_ = theta.weight.data
    beta_ans = torch.empty_like(beta_)
    theta_ans = torch.empty_like(theta_)
    for g in groups:
        group_beta = beta_[:, g]
        group_beta_shape = group_beta.shape
        group_theta = theta_[:, g]
        group_theta_shape = group_theta.shape
        group_beta, group_theta = prox(
            group_beta.reshape(-1),
            group_theta.reshape(-1),
            lambda_=select_group_value(lambda_, g, name="lambda_"),
            lambda_bar=select_group_value(lambda_bar, g, name="lambda_bar"),
            M=select_group_value(M, g, name="M"),
        )
        beta_ans[:, g] = group_beta.reshape(*group_beta_shape)
        theta_ans[:, g] = group_theta.reshape(*group_theta_shape)
    beta.weight.data, theta.weight.data = beta_ans, theta_ans
