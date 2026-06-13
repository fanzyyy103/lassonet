import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_mplconfig_dir = Path.cwd() / ".mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

from lassonet import LassoNetRegressor as OfficialLassoNetRegressor

from twostage_lassonet import TwoStagePretrainedLassoNetRegressor


def load_prsa_grouped_dataset(csv_path, *, seed=42, n_fake=10):
    rng = np.random.default_rng(seed)

    target = "pm2.5"
    group_col = "cbwd"
    base_features = [
        "year",
        "month",
        "day",
        "hour",
        "DEWP",
        "TEMP",
        "PRES",
        "Iws",
        "Is",
        "Ir",
    ]

    df = pd.read_csv(csv_path)
    df = df[[target, group_col] + base_features].dropna(
        subset=[target, group_col] + base_features
    )

    X_real = df[base_features].reset_index(drop=True)
    y = df[target].to_numpy(dtype=np.float32)
    groups_raw = df[group_col].astype(str).to_numpy()
    group_levels = sorted(pd.unique(groups_raw))
    group_to_int = {name: i for i, name in enumerate(group_levels)}
    groups = np.array([group_to_int[name] for name in groups_raw], dtype=int)

    fake_features = pd.DataFrame(
        rng.normal(size=(len(df), n_fake)),
        columns=[f"fake_{i + 1}" for i in range(n_fake)],
    )

    X_all = pd.concat([X_real, fake_features], axis=1)

    return {
        "X": X_all.to_numpy(dtype=np.float32),
        "y": y,
        "groups": groups,
        "group_levels": group_levels,
        "feature_names": list(X_all.columns),
        "group_counts": {
            key: int(value) for key, value in df[group_col].value_counts().sort_index().items()
        },
        "n_samples_clean": int(len(df)),
    }


def run_prsa_two_stage_experiment(
    csv_path,
    *,
    seed=42,
    stage1_lambda_seq=(100, 300, 1000, 3000, 10000, 30000, 100000),
    stage2_lambda_seq=(1, 3, 10, 30, 100, 300, 1000),
):
    data = load_prsa_grouped_dataset(csv_path, seed=seed)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    group_levels = data["group_levels"]
    feature_names = data["feature_names"]

    X_trainval, X_test, y_trainval, y_test, groups_trainval, groups_test = train_test_split(
        X,
        y,
        groups,
        test_size=0.2,
        random_state=seed,
        stratify=groups,
    )
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_trainval,
        y_trainval,
        groups_trainval,
        test_size=0.2,
        random_state=seed,
        stratify=groups_trainval,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    X_trainval = scaler.transform(X_trainval).astype(np.float32)

    stage1_template = OfficialLassoNetRegressor(
        hidden_dims=(10,),
        lambda_seq=list(stage1_lambda_seq),
        verbose=0,
        random_state=seed,
        torch_seed=seed,
        n_iters=(80, 30),
        patience=(10, 5),
    )
    stage1_path = stage1_template.path(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        return_state_dicts=True,
    )

    stage1_candidates = [
        {
            "lambda": float(item.lambda_),
            "selected": int(item.selected.sum().item()),
            "val_loss": float(item.val_loss),
            "state_dict": item.state_dict,
        }
        for item in stage1_path
    ]
    best_stage1 = min(stage1_candidates, key=lambda item: item["val_loss"])

    stage2_candidates = []
    for stage2_lambda in stage2_lambda_seq:
        stage1_model = OfficialLassoNetRegressor(
            hidden_dims=(10,),
            lambda_seq=[best_stage1["lambda"]],
            verbose=0,
            random_state=seed,
            torch_seed=seed,
            n_iters=(80, 30),
            patience=(10, 5),
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
                "n_iters": (80, 30),
                "patience": (10, 5),
            },
            group_model_kwargs={
                "hidden_dims": (10,),
                "dense_epochs": 60,
                "sparse_epochs": 25,
                "dense_patience": 8,
                "sparse_patience": 5,
                "random_state": seed,
                "torch_seed": seed,
                "verbose": 0,
            },
        )
        model.fit(
            X_train,
            y_train,
            groups_train,
            X_val=X_val,
            y_val=y_val,
            groups_val=groups_val,
            stage1_model=stage1_model,
            stage1_state=best_stage1["state_dict"],
        )

        pred_val = model.predict(X_val, groups_val)
        stage2_candidates.append(
            {
                "stage2_lambda": float(stage2_lambda),
                "val_mse": float(mean_squared_error(y_val, pred_val)),
                "common_support": model.get_common_support().tolist(),
                "group_support_sizes": {
                    group_levels[group]: int(len(model.get_group_final_support(group)))
                    for group in model.group_models_
                },
                "group_individual_sizes": {
                    group_levels[group]: int(
                        len(model.get_group_individual_support(group))
                    )
                    for group in model.group_models_
                },
            }
        )

    best_stage2 = min(stage2_candidates, key=lambda item: item["val_mse"])

    final_stage1 = OfficialLassoNetRegressor(
        hidden_dims=(10,),
        lambda_seq=[best_stage1["lambda"]],
        verbose=0,
        random_state=seed,
        torch_seed=seed,
        n_iters=(80, 30),
        patience=(10, 5),
    )
    final_stage1.fit(X_trainval, y_trainval)

    final_model = TwoStagePretrainedLassoNetRegressor(
        alpha=0.5,
        stage2_lambda=best_stage2["stage2_lambda"],
        common_model_kwargs={
            "hidden_dims": (10,),
            "lambda_seq": [best_stage1["lambda"]],
            "verbose": 0,
            "random_state": seed,
            "torch_seed": seed,
            "n_iters": (80, 30),
            "patience": (10, 5),
        },
        group_model_kwargs={
            "hidden_dims": (10,),
            "dense_epochs": 60,
            "sparse_epochs": 25,
            "dense_patience": 8,
            "sparse_patience": 5,
            "random_state": seed,
            "torch_seed": seed,
            "verbose": 0,
        },
    )
    final_model.fit(X_trainval, y_trainval, groups_trainval, stage1_model=final_stage1)
    pred_test = final_model.predict(X_test, groups_test)

    return {
        "n_samples_clean": data["n_samples_clean"],
        "group_counts": data["group_counts"],
        "feature_names": feature_names,
        "stage1_candidates": [
            {
                "lambda": round(item["lambda"], 6),
                "selected": item["selected"],
                "val_loss": round(item["val_loss"], 6),
            }
            for item in stage1_candidates
        ],
        "chosen_stage1": {
            "lambda": best_stage1["lambda"],
            "selected": best_stage1["selected"],
            "val_loss": round(best_stage1["val_loss"], 6),
        },
        "stage2_candidates": [
            {
                "stage2_lambda": item["stage2_lambda"],
                "val_mse": round(item["val_mse"], 6),
                "common_support_size": len(item["common_support"]),
                "group_support_sizes": item["group_support_sizes"],
                "group_individual_sizes": item["group_individual_sizes"],
            }
            for item in stage2_candidates
        ],
        "chosen_stage2": {
            "stage2_lambda": best_stage2["stage2_lambda"],
            "val_mse": round(best_stage2["val_mse"], 6),
        },
        "final_test_mse": round(float(mean_squared_error(y_test, pred_test)), 6),
        "final_common_support": final_model.get_common_support().tolist(),
        "final_common_feature_names": [
            feature_names[index] for index in final_model.get_common_support().tolist()
        ],
        "final_group_supports": {
            group_levels[group]: {
                "support_size": int(len(final_model.get_group_final_support(group))),
                "individual_size": int(
                    len(final_model.get_group_individual_support(group))
                ),
                "feature_names": [
                    feature_names[index]
                    for index in final_model.get_group_final_support(group).tolist()
                ],
                "individual_feature_names": [
                    feature_names[index]
                    for index in final_model.get_group_individual_support(group).tolist()
                ],
            }
            for group in final_model.group_models_
        },
    }


def main():
    csv_path = "/Users/zhongyangfan/Desktop/MASDS Thesis/PRSA_data_2010.1.1-2014.12.31.csv"
    results = run_prsa_two_stage_experiment(csv_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
