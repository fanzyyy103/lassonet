# Two-Stage Quantile LassoNet

## Motivation

This project extends the existing two-stage grouped LassoNet workflow to quantile regression. The goal is to keep:

- the original LassoNet architecture,
- the skip connection,
- the nonlinear branch,
- the strong hierarchy constraint,
- the existing two-stage transfer logic,

while replacing squared-error model selection with quantile check loss.

## Stage 1 Objective

For a fixed `tau in (0, 1)`, Stage 1 fits one pooled Quantile LassoNet:

`q_tau,0(x) = a_0 + x^T beta_0 + g_W0(x)`

by minimizing empirical pinball loss plus the skip-layer L1 penalty, under the unchanged LassoNet hierarchy constraint.

## Stage 2 Objective

For each group `k`, Stage 2 fits one group-specific quantile correction:

`delta_tau,k(x) = a_k + x^T beta_k + g_Wk(x)`

with a fixed Stage 1 offset:

`offset_k(alpha) = (1 - alpha) * q_hat_tau,0(x)`

Operationally, Stage 2 is fitted through the existing offset interface, equivalent to using:

`adjusted_y_k = y_k - offset_k(alpha)`

and then training a Quantile LassoNet correction on that adjusted target.

## Alpha And Support Transmission

`alpha` controls how much Stage 1 prediction information is transferred into Stage 2.

This implementation transmits:

- Stage 1 prediction offset
- Stage 1 selected support
- feature-specific penalty factors based on Stage 1 support

It does not transmit initialized weights from Stage 1 into Stage 2.

For `0 < alpha <= 1`, Stage 2 uses:

- penalty factor `1` on Stage 1-selected features
- penalty factor `1 / alpha` outside the Stage 1 support

For `alpha = 0`, Stage 2 follows the existing V2 behavior exactly: it restricts Stage 2 to the Stage 1 support columns and avoids division by zero.

## Support Types

For each group, the estimator exposes:

- Stage 2 full support: all features selected by the correction model
- Stage 2 strict new support: Stage 2 support minus Stage 1 support
- Stage 2 adjusted support: Stage 2 support that overlaps Stage 1 support
- final support: union of Stage 1 support and Stage 2 support

Stage 2 full support is not automatically interpreted as group-specific support.

## Pinball Loss

Both stages use the same `tau`, and both are trained with pinball loss:

`rho_tau(u) = max(tau * u, (tau - 1) * u)`

Validation and model selection also use pinball loss:

- Stage 1 lambda selection uses validation pinball loss
- Stage 2 lambda selection uses combined validation pinball loss
- alpha selection uses combined validation pinball loss

The estimator `score()` returns negative mean pinball loss, so larger score is better.

## Prediction Formula

For group `k`, the final prediction is:

`q_hat_tau,k(x) = (1 - alpha) * q_hat_tau,0(x) + delta_hat_tau,k(x)`

## API Example

```python
from twostage_lassonet_v2 import TwoStageQuantileLassoNetRegressor

model = TwoStageQuantileLassoNetRegressor(
    tau=0.5,
    hidden_dims_stage1=(16, 8),
    hidden_dims_stage2=(8,),
    alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    lambda_stage1=[0.001, 0.01, 0.05],
    lambda_stage2=[0.001, 0.01, 0.05],
    validation_fraction=0.2,
    random_state=42,
    refit=False,
    verbose=0,
    n_iters=(30, 12),
    patience=(6, 3),
    backtrack=True,
)

model.fit(X_train, y_train, groups_train, X_val=X_val, y_val=y_val, groups_val=groups_val)
prediction = model.predict(X_test, groups_test)
test_pinball = -model.score(X_test, y_test, groups_test)
```

## Simulation Example

The included grouped simulation uses:

- 5 groups
- 50 total features
- common features `x1` through `x10`
- exclusive group features `x11` through `x20`
- noise features `x21` through `x50`

`x3` is explicitly treated as a true common feature.

Separate runs are provided for `tau=0.5` and `tau=0.9`.

## Scope And Limitations

- The PDF directly specifies single-stage Quantile LassoNet.
- This two-stage quantile estimator is a project-specific extension built on top of the existing V2 framework.
- It inherits the tuning complexity of the original two-stage model.
- It does not reduce the number of tuning parameters automatically.
- A separate model is trained for each `tau`.
- Quantile crossing is not addressed.
- Optimization remains nonconvex.
- Stage 1 may absorb part of the group-specific signal.
- Strict Stage 2 new-support recovery may be weak even when predictive performance is reasonable.
