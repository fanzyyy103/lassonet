import numpy as np

from .two_stage import TwoStagePretrainedLassoNetRegressor


def _resolve_feature_names(feature_names, n_features):
    if feature_names is None:
        return np.asarray([f"x{index}" for index in range(n_features)], dtype=object)

    resolved = np.asarray(list(feature_names), dtype=object)
    if resolved.shape != (n_features,):
        raise ValueError(
            f"feature_names must contain exactly {n_features} entries; "
            f"received {resolved.shape[0]}."
        )
    return resolved


def _selected_mask(model):
    return np.asarray(model.selected_mask(), dtype=bool)


def _skip_importance(model):
    return (
        model.model.skip.weight.detach()
        .norm(p=2, dim=0)
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )


class PTLassoOrientedTwoStageLassoNetRegressor(TwoStagePretrainedLassoNetRegressor):
    """
    A ptLasso-oriented interface for the existing two-stage LassoNet estimator.

    Training is unchanged:
    - stage 1 learns a shared/common LassoNet on all samples;
    - stage 2 learns one group-specific residual/correction model per group.

    The difference is interpretability:
    - "common support" is the stage-1 support;
    - "group correction support" is the active stage-2 support for a group;
    - "strict group-only support" is stage-2 support minus the common support.

    This matches the ptLasso intuition more closely because a feature may be
    part of a group-specific correction even when it also appears in the
    common/shared stage-1 support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_names_ = None

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
        feature_names=None,
    ):
        n_features = np.asarray(X, dtype=np.float32).shape[1]
        self.feature_names_ = _resolve_feature_names(feature_names, n_features)
        return super().fit(
            X,
            y,
            groups,
            X_val=X_val,
            y_val=y_val,
            groups_val=groups_val,
            stage1_model=stage1_model,
            stage1_state=stage1_state,
        )

    def predict_components(self, X, groups):
        components = super().predict_components(X, groups)
        return {
            "common": components["common"],
            "group_correction": components["group_specific"],
            "group_specific": components["group_specific"],
            "final": components["final"],
        }

    def get_group_correction_support(self, group):
        return self.get_group_support(group)

    def get_group_strict_specific_support(self, group):
        return self.get_group_individual_support(group)

    def get_group_common_reused_support(self, group):
        group_support = _selected_mask(self.group_models_[group])
        return np.flatnonzero(group_support & self.common_support_)

    def get_feature_names(self):
        if self.common_support_ is None:
            raise RuntimeError("fit must be called before feature names can be read.")
        return self.feature_names_.copy()

    def summarize_group_correction(self, group, *, feature_names=None):
        if self.common_support_ is None:
            raise RuntimeError("fit must be called before supports can be summarized.")

        resolved_names = _resolve_feature_names(
            self.feature_names_ if feature_names is None else feature_names,
            len(self.common_support_),
        )
        common_mask = np.asarray(self.common_support_, dtype=bool)
        correction_mask = _selected_mask(self.group_models_[group])
        final_mask = common_mask | correction_mask
        strict_group_only_mask = correction_mask & ~common_mask
        common_reused_mask = correction_mask & common_mask

        return {
            "group": group,
            "common_support_indices": np.flatnonzero(common_mask),
            "common_support_names": resolved_names[common_mask].tolist(),
            "group_correction_indices": np.flatnonzero(correction_mask),
            "group_correction_names": resolved_names[correction_mask].tolist(),
            "strict_group_only_indices": np.flatnonzero(strict_group_only_mask),
            "strict_group_only_names": resolved_names[strict_group_only_mask].tolist(),
            "common_reused_indices": np.flatnonzero(common_reused_mask),
            "common_reused_names": resolved_names[common_reused_mask].tolist(),
            "final_union_indices": np.flatnonzero(final_mask),
            "final_union_names": resolved_names[final_mask].tolist(),
            "common_feature_count": int(common_mask.sum()),
            "group_correction_feature_count": int(correction_mask.sum()),
            "strict_group_only_count": int(strict_group_only_mask.sum()),
            "common_reused_count": int(common_reused_mask.sum()),
            "final_union_count": int(final_mask.sum()),
        }

    def get_group_correction_importance(
        self,
        group,
        *,
        feature_names=None,
        normalize=False,
    ):
        if self.common_support_ is None:
            raise RuntimeError("fit must be called before importances can be summarized.")

        resolved_names = _resolve_feature_names(
            self.feature_names_ if feature_names is None else feature_names,
            len(self.common_support_),
        )
        common_mask = np.asarray(self.common_support_, dtype=bool)
        correction_mask = _selected_mask(self.group_models_[group])
        raw_importance = _skip_importance(self.group_models_[group])

        if normalize:
            total = raw_importance.sum()
            if total > 0:
                raw_importance = raw_importance / total

        rows = []
        for index, importance in enumerate(raw_importance.tolist()):
            in_common = bool(common_mask[index])
            in_correction = bool(correction_mask[index])
            if in_common and in_correction:
                role = "common_and_group_corrected"
            elif in_correction:
                role = "group_only_correction"
            elif in_common:
                role = "common_only"
            else:
                role = "unused"

            rows.append(
                {
                    "group": group,
                    "feature_index": int(index),
                    "feature_name": str(resolved_names[index]),
                    "skip_importance": float(importance),
                    "selected_in_common": in_common,
                    "selected_in_correction": in_correction,
                    "role": role,
                }
            )

        rows.sort(key=lambda row: (-row["skip_importance"], row["feature_index"]))
        return rows

    def summarize_feature_roles(self, *, feature_names=None, only_active=True):
        if self.common_support_ is None:
            raise RuntimeError("fit must be called before feature roles can be summarized.")

        resolved_names = _resolve_feature_names(
            self.feature_names_ if feature_names is None else feature_names,
            len(self.common_support_),
        )
        common_mask = np.asarray(self.common_support_, dtype=bool)
        rows = []

        for group in self.group_order_:
            correction_mask = _selected_mask(self.group_models_[group])
            for index in range(len(common_mask)):
                in_common = bool(common_mask[index])
                in_correction = bool(correction_mask[index])
                if not in_common and not in_correction and only_active:
                    continue

                if in_common and in_correction:
                    role = "common_and_group_corrected"
                elif in_correction:
                    role = "group_only_correction"
                elif in_common:
                    role = "common_only"
                else:
                    role = "unused"

                rows.append(
                    {
                        "group": group,
                        "feature_index": int(index),
                        "feature_name": str(resolved_names[index]),
                        "role": role,
                        "selected_in_common": in_common,
                        "selected_in_correction": in_correction,
                    }
                )

        return rows

    def describe_ptlasso_structure(self, *, feature_names=None):
        if self.common_support_ is None:
            raise RuntimeError("fit must be called before structure can be described.")

        resolved_names = _resolve_feature_names(
            self.feature_names_ if feature_names is None else feature_names,
            len(self.common_support_),
        )
        summary = {
            "common_support_indices": self.get_common_support().tolist(),
            "common_support_names": resolved_names[self.common_support_].tolist(),
            "common_feature_count": int(np.asarray(self.common_support_, dtype=bool).sum()),
            "groups": {},
        }

        for group in self.group_order_:
            summary["groups"][group] = self.summarize_group_correction(
                group,
                feature_names=resolved_names,
            )

        return summary
