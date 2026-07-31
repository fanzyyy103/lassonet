# Two-Stage LassoNet V2

This package implements a separate Version 2 estimator for the thesis
formulation

```text
y_k = f_0^*(X_k) + f_k^*(X_k) + epsilon_k.
```

Version 2 keeps the original LassoNet implementation unchanged. It reuses the
local fork's `lassonet.LassoNetRegressor` path training, hierarchical proximal
projection, validation-loss history, skip-layer support, model loading, and
prediction methods.

## Model

Stage 1 fits one pooled common LassoNet:

```text
f_0(x) = mu_0 + theta_0^T x + g_W0(x)
```

The common support is the fitted skip-layer support:

```text
S_common = {j : |theta_0j| > support_tol}
```

The original package support method `model.input_mask()` is used when available;
otherwise V2 falls back to `support_tol`.

Stage 2 fits one correction LassoNet per group and per alpha candidate. For
group `k`, the frozen offset is

```text
O_k(alpha) = (1 - alpha) * f_0_hat(X_k)
```

The correction objective is implemented with the local fork's regression
`offset` support, equivalently fitting the residual target

```text
y_k - O_k(alpha).
```

For `0 < alpha <= 1`, Stage 2 uses feature-specific penalty factors

```text
pf_j(alpha) = 1        if j in S_common
pf_j(alpha) = 1/alpha  otherwise.
```

For `alpha = 0`, V2 never evaluates `1 / 0`. It creates the exact allowed
feature set `S_common` and fits the Stage 2 correction on those columns only,
then maps theta/support back to the full feature space.

The final prediction is always

```text
y_hat_k = (1 - alpha) * f_0_hat(x) + f_k_hat^corr(x).
```

## Supports

For group `k`, V2 reports:

```text
S_k          = Stage 2 correction support
S_new,k      = S_k \ S_common
S_adjusted,k = S_k intersection S_common
S_final,k    = S_common union S_k
```

`S_k` means "features used by the correction model"; it does not mean features
unique to the group.

## API

```python
from twostage_lassonet_v2 import TwoStageLassoNetRegressorV2

model = TwoStageLassoNetRegressorV2(
    hidden_dims_stage1=(64, 32),
    hidden_dims_stage2=(16,),
    M_stage1=10.0,
    M_stage2=10.0,
    alpha="auto",
    alpha_values=[0.0, 0.5, 1.0],
    lambda_stage1=[0.001, 0.01, 0.1],
    lambda_stage2=[0.001, 0.01, 0.1],
    validation_fraction=0.2,
    random_state=42,
    refit=True,
    verbose=1,
    n_iters=(100, 20),
    patience=(20, 5),
)

model.fit(X, y, groups)
y_pred = model.predict(X_test, groups_test)
components = model.predict_components(X_test, groups_test)
summary = model.get_support_summary()
tuning = model.get_tuning_summary()
```

Prediction methods:

- `predict_common(X)` returns `f_0_hat(X)`.
- `predict_offset(X, alpha=None)` returns `(1-alpha) * f_0_hat(X)`.
- `predict_correction(X, groups)` returns group correction predictions.
- `predict(X, groups)` returns final combined predictions.

## Alpha Endpoints

At `alpha = 0`, the offset is the full Stage 1 prediction and non-common
features are excluded from Stage 2 exactly through a column mask.

At `alpha = 1`, the offset is zero and every penalty factor is one, so Stage 2
reduces to separate group LassoNet fits under the same settings and seeds.

## Lambda And Alpha Selection

Stage 1 selects `lambda_0` by pooled validation MSE along the Stage 1 path.
For each alpha and group, Stage 2 selects `lambda_k(alpha)` by the combined
validation MSE:

```text
mean((y - offset - correction)^2).
```

Alpha is selected by the lowest overall combined validation MSE across groups.

## Known Limitations

- The decomposition is model-based and may not identify a unique true
  common/group-specific functional decomposition.
- The pooled Stage 1 model may capture average group-specific effects.
- Stage 2 is flexible and may relearn part of the Stage 1 signal.
- LassoNet nonlinear-only features still require nonzero skip-layer theta to
  enter the neural network through the hierarchy constraint.
- V2 does not standardize features internally; preprocess outside the estimator
  if scaling is needed.
