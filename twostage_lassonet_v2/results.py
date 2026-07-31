from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np


@dataclass
class Stage1Result:
    model: Any
    selected_lambda: float
    theta: np.ndarray
    support: np.ndarray
    path: List[Any]
    validation_scores: np.ndarray


@dataclass
class Stage2GroupResult:
    group: Any
    alpha: float
    selected_lambda: float
    correction_model: Any
    correction_theta: np.ndarray
    correction_support: np.ndarray
    new_support: np.ndarray
    adjusted_support: np.ndarray
    final_support: np.ndarray
    validation_mse: float
    path: List[Any]
    validation_loss: float | None = None


@dataclass
class AlphaCandidateResult:
    alpha: float
    validation_mse: float
    group_validation_mse: Dict[Any, float]
    group_lambdas: Dict[Any, float]
    group_results: Dict[Any, Stage2GroupResult]
    n_selected_features_by_group: Dict[Any, int]
    validation_loss: float | None = None
    group_validation_loss: Dict[Any, float] | None = None


@dataclass
class TwoStageFitResult:
    stage1: Stage1Result
    alpha_results: List[AlphaCandidateResult]
    selected_alpha: float
    selected_stage2: Mapping[Any, Stage2GroupResult]


@dataclass
class PredictionComponents:
    common_prediction: np.ndarray
    scaled_common_offset: np.ndarray
    correction_prediction: np.ndarray
    final_prediction: np.ndarray
