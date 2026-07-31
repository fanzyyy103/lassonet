from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_BASE_COEFFICIENTS = np.array(
    [2.0, -1.5, 1.2, -1.0, 0.8, -0.7, 1.5, -1.3, 1.0, -0.9],
    dtype=np.float64,
)

DEFAULT_GROUP_NAMES = {index: f"Group {index + 1}" for index in range(5)}
GROUPED_COMMON_FORMULA = (
    "sqrt(x1^2 + x2^3 + x3^2 + sqrt(x4^2 + x5^2 + x6^2 + x7^2 + x8^2 + x9^2 + x10^2))"
)

COMMON_SUPPORT_INDICES = np.arange(0, 10, dtype=np.int64)
NONSHARED_SUPPORT_INDICES = np.arange(10, 20, dtype=np.int64)
NOISE_SUPPORT_INDICES = np.arange(20, 50, dtype=np.int64)


@dataclass(frozen=True)
class GroupedSyntheticData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: tuple[str, ...]
    group_names: dict[int, str]
    common_signal: np.ndarray
    group_specific_signal: np.ndarray
    noise: np.ndarray
    group_coefficients: np.ndarray
    base_coefficients: np.ndarray
    coefficient_perturbations: np.ndarray
    common_support_indices: np.ndarray
    nonshared_support_indices: np.ndarray
    noise_support_indices: np.ndarray
    common_support_names: tuple[str, ...]
    nonshared_support_names: tuple[str, ...]
    noise_support_names: tuple[str, ...]
    group_row_indices: dict[int, np.ndarray]
    seed: int | None
    feature_seed: int | None
    coefficient_seed: int | None
    noise_seed: int | None
    noise_sd: float
    coefficient_perturbation_sd: float
    common_formula: str = GROUPED_COMMON_FORMULA


@dataclass(frozen=True)
class GroupedSyntheticSplit:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class FeatureScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_


@dataclass(frozen=True)
class GroupedSyntheticSplits:
    train: GroupedSyntheticSplit
    validation: GroupedSyntheticSplit
    test: GroupedSyntheticSplit
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    scaler: FeatureScaler | None = None


def _as_float64_array(values: np.ndarray | list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return array


def _validate_base_coefficients(base_coefficients: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    base_array = _as_float64_array(base_coefficients, name="base_coefficients")
    if base_array.shape[0] != 10:
        raise ValueError("base_coefficients must contain exactly 10 values for x11 through x20.")
    return base_array


def _validate_positive_integer(value: int, *, name: str) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _seed_from_sequence(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _make_feature_names(n_features: int) -> tuple[str, ...]:
    return tuple(f"x{index}" for index in range(1, n_features + 1))


def compute_common_signal(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 10:
        raise ValueError("X must be a two-dimensional array with at least 10 columns.")

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]
    x5 = X[:, 4]
    x6 = X[:, 5]
    x7 = X[:, 6]
    x8 = X[:, 7]
    x9 = X[:, 8]
    x10 = X[:, 9]

    inner_radicand = x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2 + x10**2
    if np.min(inner_radicand) < -1e-12:
        raise ValueError("The inner common-function radicand became negative.")
    inner_root = np.sqrt(np.clip(inner_radicand, a_min=0.0, a_max=None))

    outer_radicand = x1**2 + x2**3 + x3**2 + inner_root
    if np.min(outer_radicand) < -1e-12:
        raise ValueError(
            "The outer common-function radicand became negative. "
            "This indicates an invalid feature distribution or implementation error."
        )
    return np.sqrt(np.clip(outer_radicand, a_min=0.0, a_max=None))


def generate_grouped_fake_dataset(
    *,
    n_per_group: int = 1000,
    n_groups: int = 5,
    n_features: int = 50,
    noise_sd: float = 0.5,
    coefficient_perturbation_sd: float = 0.1,
    base_coefficients: np.ndarray | list[float] | tuple[float, ...] = DEFAULT_BASE_COEFFICIENTS,
    random_state: int | None = 123,
) -> GroupedSyntheticData:
    _validate_positive_integer(n_per_group, name="n_per_group")
    _validate_positive_integer(n_groups, name="n_groups")
    _validate_positive_integer(n_features, name="n_features")

    if n_groups != 5:
        raise ValueError("This implementation currently supports exactly 5 groups.")
    if n_features != 50:
        raise ValueError("This implementation currently supports exactly 50 features.")
    if noise_sd < 0.0:
        raise ValueError("noise_sd must be nonnegative.")
    if coefficient_perturbation_sd < 0.0:
        raise ValueError("coefficient_perturbation_sd must be nonnegative.")

    base_array = _validate_base_coefficients(base_coefficients)

    seed_sequence = np.random.SeedSequence(random_state)
    feature_sequence, coefficient_sequence, noise_sequence = seed_sequence.spawn(3)
    feature_seed = _seed_from_sequence(feature_sequence)
    coefficient_seed = _seed_from_sequence(coefficient_sequence)
    noise_seed = _seed_from_sequence(noise_sequence)

    feature_rng = np.random.default_rng(feature_seed)
    coefficient_rng = np.random.default_rng(coefficient_seed)
    noise_rng = np.random.default_rng(noise_seed)

    n_samples = n_per_group * n_groups
    X_common = feature_rng.uniform(0.0, 1.0, size=(n_samples, 10))
    X_remaining = feature_rng.normal(loc=0.0, scale=1.0, size=(n_samples, 40))
    X = np.hstack([X_common, X_remaining]).astype(np.float64, copy=False)

    groups = np.repeat(np.arange(n_groups, dtype=np.int64), n_per_group)

    coefficient_perturbations = np.zeros((n_groups, 10), dtype=np.float64)
    coefficient_perturbations[1:, :] = coefficient_rng.normal(
        loc=0.0,
        scale=coefficient_perturbation_sd,
        size=(n_groups - 1, 10),
    )
    group_coefficients = base_array[None, :] + coefficient_perturbations

    common_signal = compute_common_signal(X)
    group_specific_signal = np.sum(X[:, 10:20] * group_coefficients[groups], axis=1)
    noise = noise_rng.normal(loc=0.0, scale=noise_sd, size=n_samples)
    y = common_signal + group_specific_signal + noise

    feature_names = _make_feature_names(n_features)
    common_support_names = tuple(feature_names[index] for index in COMMON_SUPPORT_INDICES)
    nonshared_support_names = tuple(feature_names[index] for index in NONSHARED_SUPPORT_INDICES)
    noise_support_names = tuple(feature_names[index] for index in NOISE_SUPPORT_INDICES)
    group_row_indices = {
        group: np.flatnonzero(groups == group).astype(np.int64, copy=False)
        for group in range(n_groups)
    }

    return GroupedSyntheticData(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        group_names=DEFAULT_GROUP_NAMES.copy(),
        common_signal=common_signal,
        group_specific_signal=group_specific_signal,
        noise=noise,
        group_coefficients=group_coefficients,
        base_coefficients=base_array.copy(),
        coefficient_perturbations=coefficient_perturbations,
        common_support_indices=COMMON_SUPPORT_INDICES.copy(),
        nonshared_support_indices=NONSHARED_SUPPORT_INDICES.copy(),
        noise_support_indices=NOISE_SUPPORT_INDICES.copy(),
        common_support_names=common_support_names,
        nonshared_support_names=nonshared_support_names,
        noise_support_names=noise_support_names,
        group_row_indices=group_row_indices,
        seed=None if random_state is None else int(random_state),
        feature_seed=feature_seed,
        coefficient_seed=coefficient_seed,
        noise_seed=noise_seed,
        noise_sd=float(noise_sd),
        coefficient_perturbation_sd=float(coefficient_perturbation_sd),
    )


def validate_grouped_fake_dataset(
    data: GroupedSyntheticData,
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    X = np.asarray(data.X, dtype=np.float64)
    y = np.asarray(data.y, dtype=np.float64)
    groups = np.asarray(data.groups, dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != 50:
        raise ValueError("X must have shape (n_samples, 50).")
    if y.shape != (X.shape[0],):
        raise ValueError("y must have shape (n_samples,).")
    if groups.shape != (X.shape[0],):
        raise ValueError("groups must have shape (n_samples,).")
    if tuple(sorted(data.group_names)) != (0, 1, 2, 3, 4):
        raise ValueError("group_names must provide the mapping for labels 0 through 4.")
    if "x3" not in data.common_support_names:
        raise ValueError("x3 must be included in the common support.")
    if "x3" in data.noise_support_names:
        raise ValueError("x3 must never be classified as noise.")

    expected_common = compute_common_signal(X)
    if not np.allclose(expected_common, data.common_signal, atol=atol, rtol=0.0):
        raise ValueError("common_signal does not match the specified common nonlinear function.")

    expected_group_signal = np.sum(X[:, 10:20] * data.group_coefficients[groups], axis=1)
    if not np.allclose(expected_group_signal, data.group_specific_signal, atol=atol, rtol=0.0):
        raise ValueError("group_specific_signal does not match the group coefficients.")

    if not np.allclose(data.group_coefficients[0], data.base_coefficients, atol=atol, rtol=0.0):
        raise ValueError("Group 1 coefficients must equal the base coefficients.")
    if not np.allclose(data.coefficient_perturbations[0], 0.0, atol=atol, rtol=0.0):
        raise ValueError("Group 1 perturbations must be exactly zero.")
    if not np.allclose(
        data.group_coefficients,
        data.base_coefficients[None, :] + data.coefficient_perturbations,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("group_coefficients must equal base_coefficients + coefficient_perturbations.")

    reconstructed_y = data.common_signal + data.group_specific_signal + data.noise
    if not np.allclose(reconstructed_y, y, atol=atol, rtol=0.0):
        raise ValueError("y must equal common_signal + group_specific_signal + noise.")

    support_union = np.concatenate(
        [
            data.common_support_indices,
            data.nonshared_support_indices,
            data.noise_support_indices,
        ]
    )
    if set(support_union.tolist()) != set(range(50)):
        raise ValueError("Support partitions must cover x1 through x50 exactly once.")

    unique_groups, counts = np.unique(groups, return_counts=True)
    if not np.array_equal(unique_groups, np.arange(5, dtype=np.int64)):
        raise ValueError("groups must contain labels 0 through 4.")

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "group_counts": {int(group): int(count) for group, count in zip(unique_groups, counts)},
        "common_support_names": list(data.common_support_names),
        "nonshared_support_names": list(data.nonshared_support_names),
        "noise_support_names": list(data.noise_support_names),
        "x3_is_common": True,
        "x21_to_x50_zero_true_coefficients": True,
        "max_abs_group_coefficient_deviation_from_group1": float(
            np.max(np.abs(data.group_coefficients - data.group_coefficients[0]))
        ),
        "max_abs_response_residual": float(np.max(np.abs(reconstructed_y - y))),
        "common_formula": data.common_formula,
    }


def fit_feature_scaler(X_train: np.ndarray) -> FeatureScaler:
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.ndim != 2:
        raise ValueError("X_train must be a two-dimensional array.")
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return FeatureScaler(mean_=mean, scale_=scale)


def split_grouped_fake_dataset(
    data: GroupedSyntheticData,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int = 123,
    scale: bool = False,
) -> GroupedSyntheticSplits:
    if min(train_fraction, validation_fraction, test_fraction) < 0.0:
        raise ValueError("Split fractions must be nonnegative.")
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    rng = np.random.default_rng(random_state)
    train_indices_parts: list[np.ndarray] = []
    validation_indices_parts: list[np.ndarray] = []
    test_indices_parts: list[np.ndarray] = []

    for group in sorted(data.group_names):
        group_indices = np.asarray(data.group_row_indices[group], dtype=np.int64)
        shuffled = rng.permutation(group_indices)
        n_group = shuffled.shape[0]
        n_train = int(np.floor(train_fraction * n_group))
        n_validation = int(np.floor(validation_fraction * n_group))
        n_test = n_group - n_train - n_validation
        if n_test < 0:
            raise ValueError("Invalid split fractions produced a negative test size.")

        train_indices_parts.append(shuffled[:n_train])
        validation_indices_parts.append(shuffled[n_train : n_train + n_validation])
        test_indices_parts.append(shuffled[n_train + n_validation : n_train + n_validation + n_test])

    train_indices = np.concatenate(train_indices_parts)
    validation_indices = np.concatenate(validation_indices_parts)
    test_indices = np.concatenate(test_indices_parts)

    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)
    test_indices = rng.permutation(test_indices)

    train = GroupedSyntheticSplit(
        X=np.asarray(data.X[train_indices], dtype=np.float64),
        y=np.asarray(data.y[train_indices], dtype=np.float64),
        groups=np.asarray(data.groups[train_indices], dtype=np.int64),
        indices=train_indices,
    )
    validation = GroupedSyntheticSplit(
        X=np.asarray(data.X[validation_indices], dtype=np.float64),
        y=np.asarray(data.y[validation_indices], dtype=np.float64),
        groups=np.asarray(data.groups[validation_indices], dtype=np.int64),
        indices=validation_indices,
    )
    test = GroupedSyntheticSplit(
        X=np.asarray(data.X[test_indices], dtype=np.float64),
        y=np.asarray(data.y[test_indices], dtype=np.float64),
        groups=np.asarray(data.groups[test_indices], dtype=np.int64),
        indices=test_indices,
    )

    scaler = None
    if scale:
        scaler = fit_feature_scaler(train.X)
        train = GroupedSyntheticSplit(
            X=scaler.transform(train.X),
            y=train.y,
            groups=train.groups,
            indices=train.indices,
        )
        validation = GroupedSyntheticSplit(
            X=scaler.transform(validation.X),
            y=validation.y,
            groups=validation.groups,
            indices=validation.indices,
        )
        test = GroupedSyntheticSplit(
            X=scaler.transform(test.X),
            y=test.y,
            groups=test.groups,
            indices=test.indices,
        )

    return GroupedSyntheticSplits(
        train=train,
        validation=validation,
        test=test,
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
        test_fraction=float(test_fraction),
        scaler=scaler,
    )


def validate_grouped_fake_splits(
    data: GroupedSyntheticData,
    splits: GroupedSyntheticSplits,
) -> dict[str, Any]:
    train_set = set(np.asarray(splits.train.indices, dtype=np.int64).tolist())
    validation_set = set(np.asarray(splits.validation.indices, dtype=np.int64).tolist())
    test_set = set(np.asarray(splits.test.indices, dtype=np.int64).tolist())

    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise ValueError("Train/validation/test splits must be disjoint.")

    all_indices = train_set | validation_set | test_set
    expected_indices = set(range(data.X.shape[0]))
    if all_indices != expected_indices:
        raise ValueError("Train/validation/test splits must cover the full dataset exactly once.")

    split_counts = {}
    for split_name, split in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        groups, counts = np.unique(split.groups, return_counts=True)
        split_counts[split_name] = {int(group): int(count) for group, count in zip(groups, counts)}

    return {
        "split_shapes": {
            "train": tuple(splits.train.X.shape),
            "validation": tuple(splits.validation.X.shape),
            "test": tuple(splits.test.X.shape),
        },
        "split_group_counts": split_counts,
        "scaler_fitted_on_train_only": splits.scaler is not None,
        "all_indices_covered": True,
        "no_overlap": True,
    }
