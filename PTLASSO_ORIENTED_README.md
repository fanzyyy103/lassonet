# ptLasso-Oriented Two-Stage LassoNet

This note documents the ptLasso-oriented interpretation layer added on top of the
existing two-stage LassoNet package.

## Why This Version Exists

The original `TwoStagePretrainedLassoNetRegressor` already follows the main
two-stage ptLasso-style transfer logic:

1. Stage 1 fits a shared/common LassoNet on all samples.
2. Stage 2 fits one residual LassoNet per group.
3. Stage 2 uses:
   - the stage-1 prediction as an offset term;
   - the stage-1 support to build feature-wise penalty factors;
   - `alpha` to control how strongly stage-1 information is transferred.

The issue was interpretability.

The earlier experiment scripts treated a feature as "group-specific" only when:

- it was selected in stage 2; and
- it was not selected in stage 1.

That is a strict set-difference definition:

```text
strict_group_only = stage2_support - stage1_common_support
```

This is often too narrow for ptLasso-style grouped modeling because a feature can:

- belong to the shared/common stage-1 support; and
- still be actively reused by the group-specific stage-2 correction.

For seasonal or directional datasets, this is often the realistic case.

## New Class

Use:

```python
from twostage_lassonet import PTLassoOrientedTwoStageLassoNetRegressor
```

This class keeps the same training logic as the original two-stage estimator, but
adds a more ptLasso-oriented interpretation API.

## Core Interpretation Layers

For each group, the new class separates features into three layers:

1. `common support`
   - the stage-1 support shared across groups

2. `group correction support`
   - the active support of the stage-2 model for a given group
   - this is the main ptLasso-oriented notion of group-specific correction
   - it may overlap with common support

3. `strict group-only support`
   - features that appear in stage 2 but not in stage 1
   - this is still useful, but it is only one special case of group correction

## Available Methods

### Shared/common support

- `get_common_support()`
- `get_feature_names()`

### Group correction support

- `get_group_correction_support(group)`
- `get_group_common_reused_support(group)`
- `get_group_strict_specific_support(group)`
- `get_group_final_support(group)`

### Group correction summaries

- `summarize_group_correction(group)`
- `describe_ptlasso_structure()`
- `summarize_feature_roles()`

### Group correction importance

- `get_group_correction_importance(group)`

This returns one row per feature with:

- `skip_importance`
- whether the feature is in the common support
- whether the feature is active in the group correction
- a role label:
  - `common_only`
  - `common_and_group_corrected`
  - `group_only_correction`
  - `unused`

## Prediction Decomposition

`predict_components(X, groups)` now returns:

- `common`
- `group_correction`
- `group_specific`
- `final`

`group_correction` and `group_specific` are aliases for the same stage-2 residual
component. The extra `group_correction` name is included because it aligns better
with the ptLasso interpretation.

## Example

```python
import numpy as np

from twostage_lassonet import PTLassoOrientedTwoStageLassoNetRegressor

X = np.random.randn(200, 8).astype("float32")
groups = np.repeat([0, 1], 100)
y = (
    2.0 * X[:, 0]
    - 1.5 * X[:, 1]
    + (groups == 0) * 2.0 * X[:, 2]
    + (groups == 1) * 2.0 * X[:, 3]
).astype("float32")

model = PTLassoOrientedTwoStageLassoNetRegressor(
    alpha=0.5,
    stage1_lambda=1e-2,
    stage2_lambda=1e-2,
    common_model_kwargs={"hidden_dims": (16,), "dense_epochs": 50, "sparse_epochs": 30},
    group_model_kwargs={"hidden_dims": (16,), "dense_epochs": 50, "sparse_epochs": 30},
)

model.fit(
    X,
    y,
    groups,
    feature_names=[f"x{i}" for i in range(X.shape[1])],
)

print("Common support:", model.get_common_support())
print("Group 0 correction support:", model.get_group_correction_support(0))
print("Group 0 strict group-only support:", model.get_group_strict_specific_support(0))
print("Group 0 summary:", model.summarize_group_correction(0))
```

## Recommended Experimental Use

When reporting grouped experiments, it is helpful to distinguish:

1. Stage-1 common features
2. Stage-2 correction features by group
3. Strictly new group-only features
4. Common features that are also reused in group corrections

This is usually more faithful to the ptLasso idea than reporting only
`stage2_support - stage1_support`.
