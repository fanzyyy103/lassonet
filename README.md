# Two-Stage Pretrained LassoNet

This repository implements the two-stage grouped-sample formulation from `thesis_formula.pdf` using PyTorch and a LassoNet-style hierarchy constraint.

The model follows the document closely:

1. Stage 1 trains one common LassoNet on all observations:
   `f_0(x) = mu_0 + theta_0^T x + g_{W_0}(x)`.
2. Stage 2 trains one group-specific LassoNet per sample group `k`:
   `f_k(x) = mu_k + theta_k^T x + g_{W_k}(x)`.
3. The final prediction for group `k` is:
   `y_hat_k = (1 - alpha) * f_hat_0(X_k) + f_hat_k(X_k)`.
4. The second stage uses the stage-1 support `S_hat_0` to define
   the penalty factor:
   `pf_j = 1` if `j in S_hat_0`, otherwise `pf_j = 1 / alpha`.

## Files

- [twostage_lassonet/prox.py](</Users/zhongyangfan/Documents/lassonet 2/twostage_lassonet/prox.py>): proximal operator adapted from `lasso-net/lassonet`
- [twostage_lassonet/model.py](</Users/zhongyangfan/Documents/lassonet 2/twostage_lassonet/model.py>): LassoNet network
- [twostage_lassonet/regression.py](</Users/zhongyangfan/Documents/lassonet 2/twostage_lassonet/regression.py>): single-model regression trainer
- [twostage_lassonet/two_stage.py](</Users/zhongyangfan/Documents/lassonet 2/twostage_lassonet/two_stage.py>): two-stage grouped-feature trainer

## Quick Start

```python
import numpy as np

from twostage_lassonet import TwoStagePretrainedLassoNetRegressor

X = np.random.randn(200, 20).astype("float32")
groups = np.repeat([0, 1], 100)
y = (
    2.0 * X[:, 0]
    - 1.5 * X[:, 1]
    + (groups == 0) * 2.5 * X[:, 2]
    + (groups == 1) * 2.5 * X[:, 3]
).astype("float32")

model = TwoStagePretrainedLassoNetRegressor(
    alpha=0.5,
    stage1_lambda=1e-2,
    stage2_lambda=1e-2,
    common_model_kwargs={
        "hidden_dims": (16,),
        "dense_epochs": 100,
        "sparse_epochs": 80,
        "verbose": 0,
    },
    group_model_kwargs={
        "hidden_dims": (16,),
        "dense_epochs": 100,
        "sparse_epochs": 80,
        "verbose": 0,
    },
)

model.fit(X, y, groups)
pred = model.predict(X, groups)

print("common support:", model.get_common_support())
print("group 0 final support:", model.get_group_final_support(0))
print("group 1 final support:", model.get_group_final_support(1))
```

For a ptLasso-oriented interpretation layer that separates:

- stage-1 common support
- stage-2 group correction support
- strict group-only support

use:

```python
from twostage_lassonet import PTLassoOrientedTwoStageLassoNetRegressor
```

See [PTLASSO_ORIENTED_README.md](PTLASSO_ORIENTED_README.md) for details.

## Generate The Three Path Plots

The repository also includes code to generate the three stacked plots:

1. `score` vs `number of selected features`
2. `score` vs `lambda`
3. `number of selected features` vs `lambda`

Run:

```bash
cd "/Users/zhongyangfan/Documents/lassonet 2"
PYTHONPATH=. python3 examples/plot_lambda_path.py
```

The figure will be saved to:

```text
outputs/lambda_path_summary.png
```
