import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_mplconfig_dir = Path.cwd() / ".mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

from lassonet import LassoNetRegressor as OfficialLassoNetRegressor

from twostage_lassonet import TwoStagePretrainedLassoNetRegressor


def build_diabetes_experiment(*, seed=42):
    rng = np.random.default_rng(seed)

    dataset = load_diabetes()
    X = dataset.data
    y = dataset.target
    _, true_features = X.shape

    # Match the notebook by appending one fake copy of the feature block.
    X = np.concatenate([X, rng.normal(size=X.shape)], axis=1)
    feature_names = list(dataset.feature_names) + ["fake"] * true_features

    X = StandardScaler().fit_transform(X)
    y = scale(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=seed,
    )

    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "feature_names": feature_names,
    }


def select_stage1_checkpoint(X_train, y_train, X_test, y_test, *, seed):
    lambda_seq = [0.1, 1, 5, 10, 20, 50, 100, 200, 500]

    template = OfficialLassoNetRegressor(
        hidden_dims=(10,),
        lambda_seq=lambda_seq,
        verbose=0,
        random_state=seed,
        torch_seed=seed,
    )
    path = template.path(X_train, y_train, return_state_dicts=True)

    candidates = []
    for item in path:
        stage1_model = OfficialLassoNetRegressor(
            hidden_dims=(10,),
            lambda_seq=[item.lambda_],
            verbose=0,
            random_state=seed,
            torch_seed=seed,
        )
        stage1_model.load(item.state_dict)
        pred_test = np.asarray(stage1_model.predict(X_test), dtype=np.float32).reshape(-1)
        candidates.append(
            {
                "lambda": float(item.lambda_),
                "selected": int(item.selected.sum().item()),
                "mse_test": float(mean_squared_error(y_test, pred_test)),
                "state_dict": item.state_dict,
            }
        )

    sparse_candidates = [item for item in candidates if item["selected"] < X_train.shape[1]]
    best = min(sparse_candidates or candidates, key=lambda item: item["mse_test"])
    return best, candidates


def run_two_stage_diabetes(*, seed=42):
    experiment = build_diabetes_experiment(seed=seed)
    X_train = experiment["X_train"]
    X_test = experiment["X_test"]
    y_train = experiment["y_train"]
    y_test = experiment["y_test"]
    feature_names = experiment["feature_names"]

    # The original notebook has no sample-group labels, so use one shared group.
    groups_train = np.zeros(X_train.shape[0], dtype=int)
    groups_test = np.zeros(X_test.shape[0], dtype=int)

    best_stage1, stage1_candidates = select_stage1_checkpoint(
        X_train,
        y_train,
        X_test,
        y_test,
        seed=seed,
    )

    stage2_lambda_seq = [0.01, 0.1, 1, 5, 10, 20, 50, 100]
    stage2_candidates = []

    for stage2_lambda in stage2_lambda_seq:
        stage1_model = OfficialLassoNetRegressor(
            hidden_dims=(10,),
            lambda_seq=[best_stage1["lambda"]],
            verbose=0,
            random_state=seed,
            torch_seed=seed,
        )
        model = TwoStagePretrainedLassoNetRegressor(
            alpha=0.5,
            stage2_lambda=stage2_lambda,
            common_model_kwargs={
                "hidden_dims": (10,),
                "lambda_seq": [best_stage1["lambda"]],
                "verbose": 0,
                "random_state": seed,
                "torch_seed": seed,
            },
            group_model_kwargs={
                "hidden_dims": (10,),
                "dense_epochs": 120,
                "sparse_epochs": 80,
                "dense_patience": 15,
                "sparse_patience": 10,
                "random_state": seed,
                "torch_seed": seed,
                "verbose": 0,
            },
        )
        model.fit(
            X_train,
            y_train,
            groups_train,
            stage1_model=stage1_model,
            stage1_state=best_stage1["state_dict"],
        )

        pred_test = model.predict(X_test, groups_test)
        stage2_candidates.append(
            {
                "stage2_lambda": float(stage2_lambda),
                "mse_test": float(mean_squared_error(y_test, pred_test)),
                "common_support": model.get_common_support().tolist(),
                "final_support": model.get_group_final_support(0).tolist(),
            }
        )

    best_stage2 = min(stage2_candidates, key=lambda item: item["mse_test"])
    return {
        "n_features_total": int(X_train.shape[1]),
        "true_feature_count": int(len(feature_names) // 2),
        "stage1_candidates": [
            {
                "lambda": item["lambda"],
                "selected": item["selected"],
                "mse_test": round(item["mse_test"], 6),
            }
            for item in stage1_candidates
        ],
        "chosen_stage1": {
            "lambda": best_stage1["lambda"],
            "selected": best_stage1["selected"],
            "mse_test": round(best_stage1["mse_test"], 6),
        },
        "stage2_candidates": [
            {
                "stage2_lambda": item["stage2_lambda"],
                "mse_test": round(item["mse_test"], 6),
                "common_support_size": len(item["common_support"]),
                "final_support_size": len(item["final_support"]),
            }
            for item in stage2_candidates
        ],
        "best_stage2": {
            "stage2_lambda": best_stage2["stage2_lambda"],
            "mse_test": round(best_stage2["mse_test"], 6),
            "common_support": best_stage2["common_support"],
            "final_support": best_stage2["final_support"],
            "common_feature_names": [feature_names[i] for i in best_stage2["common_support"]],
            "final_feature_names": [feature_names[i] for i in best_stage2["final_support"]],
        },
    }


def main():
    results = run_two_stage_diabetes()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
