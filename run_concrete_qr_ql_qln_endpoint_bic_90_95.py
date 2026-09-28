from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMPDIR / "concrete_qr_ql_qln_endpoint_bic_90_95_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt

import run_quantile_lassonet_concrete_xls_experiment as concrete
import run_real_datasets_normalized_pi_aware_bic as real_normalized
from quantile_lassonet_bic_implementation import fit_quantile_lassonet_bic
from twostage_lassonet_v2.utils import mean_pinball_loss


OUTPUT_DIR = Path("outputs") / "concrete_qr_ql_qln_endpoint_bic_90_95"
ENDPOINT_TAUS = (0.025, 0.05, 0.95, 0.975)
INTERVALS = [
    {"Interval": "95%", "Lower Tau": 0.025, "Upper Tau": 0.975, "Target": 0.95},
    {"Interval": "90%", "Lower Tau": 0.05, "Upper Tau": 0.95, "Target": 0.90},
]
MODEL_ORDER = ["QR", "QL", "QLN Endpoint-BIC"]


def round_frame(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(digits)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    display = round_frame(frame).fillna("")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def selected_feature_text(mask: np.ndarray, feature_names: np.ndarray) -> str:
    return concrete.selected_feature_string(np.asarray(mask, dtype=bool), feature_names)


def interval_row(
    *,
    model: str,
    interval_spec: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    widths = upper - lower
    covered = (lower <= y_test) & (y_test <= upper)
    return {
        "Dataset": "Concrete",
        "Interval": interval_spec["Interval"],
        "Model": model,
        "Target": float(interval_spec["Target"]),
        "Covered Count": int(np.sum(covered)),
        "Test N": int(len(y_test)),
        "PI Coverage": float(np.mean(covered)),
        "Coverage Error": abs(float(np.mean(covered)) - float(interval_spec["Target"])),
        "Signed Coverage Error": float(np.mean(covered)) - float(interval_spec["Target"]),
        "Lower Miss Count": int(np.sum(y_test < lower)),
        "Upper Miss Count": int(np.sum(y_test > upper)),
        "Mean PI Length": float(np.mean(widths)),
        "Median PI Length": float(np.median(widths)),
        "Std PI Length": float(np.std(widths, ddof=0)),
        "Min PI Length": float(np.min(widths)),
        "Max PI Length": float(np.max(widths)),
        "Crossing Rate": float(np.mean(lower > upper)),
    }


def plot_comparison(interval_frame: pd.DataFrame, output_path: Path) -> None:
    frame = interval_frame.copy()
    frame["Model Order"] = frame["Model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
    for row_index, interval in enumerate(["95%", "90%"]):
        subset = frame[frame["Interval"] == interval].sort_values("Model Order")
        x = np.arange(len(subset), dtype=np.float64)
        axes[row_index, 0].bar(x, subset["PI Coverage"], color="#4C78A8")
        axes[row_index, 0].axhline(float(subset["Target"].iloc[0]), color="crimson", linestyle="--", linewidth=1.2)
        axes[row_index, 0].set_xticks(x)
        axes[row_index, 0].set_xticklabels(subset["Model"], rotation=12, ha="right")
        axes[row_index, 0].set_ylabel("PI coverage")
        axes[row_index, 0].set_title(f"{interval} coverage")
        axes[row_index, 1].bar(x, subset["Mean PI Length"], color="#F58518")
        axes[row_index, 1].set_xticks(x)
        axes[row_index, 1].set_xticklabels(subset["Model"], rotation=12, ha="right")
        axes[row_index, 1].set_ylabel("Mean PI length")
        axes[row_index, 1].set_title(f"{interval} mean length")
    fig.suptitle("Concrete: QR / QL / QLN endpoint-wise BIC", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = real_normalized.load_concrete()
    feature_names = np.asarray(dataset["feature_names"], dtype=object)
    concrete.set_all_seeds(concrete.SEED)

    predictions: dict[str, dict[float, np.ndarray]] = {"QR": {}, "QL": {}, "QLN Endpoint-BIC": {}}
    point_rows: list[dict[str, Any]] = []
    ql_path_rows: list[dict[str, Any]] = []
    qln_path_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for tau in ENDPOINT_TAUS:
        print(f"[QR] tau={tau}", flush=True)
        qr = concrete.fit_standard_quantile_regression(dataset["X_train"], dataset["y_train"], dataset["X_test"], tau)
        predictions["QR"][tau] = qr["prediction"]
        selected_rows.append(
            {
                "Model": "QR",
                "Tau": tau,
                "Selected Parameter": qr["alpha_used"],
                "Selected Count": pd.NA,
                "Selected Features": pd.NA,
            }
        )

        print(f"[QL] tau={tau}", flush=True)
        ql = concrete.fit_quantile_lasso(
            dataset["X_train"],
            dataset["y_train"],
            dataset["X_val"],
            dataset["y_val"],
            dataset["X_test"],
            tau,
            feature_names,
        )
        predictions["QL"][tau] = ql["prediction"]
        ql_path_rows.extend(ql["path_rows"])
        selected_rows.append(
            {
                "Model": "QL",
                "Tau": tau,
                "Selected Parameter": ql["best_alpha"],
                "Selected Count": int(np.sum(ql["support_mask"])),
                "Selected Features": selected_feature_text(ql["support_mask"], feature_names),
            }
        )

        print(f"[QLN Endpoint-BIC] tau={tau}", flush=True)
        dataset["set_all_seeds"](int(dataset["seed"]))
        qln = fit_quantile_lassonet_bic(
            X_train=dataset["X_train"],
            y_train=dataset["y_train"],
            X_val=dataset["X_val"],
            y_val=dataset["y_val"],
            X_test=dataset["X_test"],
            tau=tau,
            lassonet_config=dataset["qln_config"],
            feature_names=feature_names,
        )
        predictions["QLN Endpoint-BIC"][tau] = qln.prediction
        qln_path_rows.extend(qln.path_rows)
        selected_rows.append(
            {
                "Model": "QLN Endpoint-BIC",
                "Tau": tau,
                "Selected Parameter": qln.best_lambda,
                "Selected Count": qln.n_selected,
                "Selected Features": selected_feature_text(qln.support_mask, feature_names),
                "BIC": qln.best_bic,
                "Validation Pinball": qln.validation_pinball_at_selected_lambda,
            }
        )

        for model in MODEL_ORDER:
            prediction = predictions[model][tau]
            point_rows.append(
                {
                    "Dataset": "Concrete",
                    "Model": model,
                    "Tau": tau,
                    "Test Pinball": mean_pinball_loss(dataset["y_test"], prediction, tau),
                    "MAE": float(np.mean(np.abs(dataset["y_test"] - prediction))),
                }
            )

    interval_rows = []
    for spec in INTERVALS:
        lower_tau = float(spec["Lower Tau"])
        upper_tau = float(spec["Upper Tau"])
        for model in MODEL_ORDER:
            interval_rows.append(
                interval_row(
                    model=model,
                    interval_spec=spec,
                    lower=predictions[model][lower_tau],
                    upper=predictions[model][upper_tau],
                    y_test=dataset["y_test"],
                )
            )
    interval_frame = pd.DataFrame(interval_rows)
    interval_frame["Coverage Error Rank"] = interval_frame.groupby("Interval")["Coverage Error"].rank(
        method="min", ascending=True
    )
    interval_frame["Length Rank"] = interval_frame.groupby("Interval")["Mean PI Length"].rank(
        method="min", ascending=True
    )
    point_frame = pd.DataFrame(point_rows)
    selected_frame = pd.DataFrame(selected_rows)
    ql_path_frame = pd.DataFrame(ql_path_rows)
    qln_path_frame = pd.DataFrame(qln_path_rows)

    plot_path = OUTPUT_DIR / "concrete_qr_ql_qln_endpoint_bic_90_95.png"
    plot_comparison(interval_frame, plot_path)

    paths = {
        "interval": OUTPUT_DIR / "concrete_pi_coverage_length_qr_ql_qln_endpoint_bic.csv",
        "point": OUTPUT_DIR / "concrete_point_metrics_qr_ql_qln_endpoint_bic.csv",
        "selected": OUTPUT_DIR / "concrete_selected_parameters_qr_ql_qln_endpoint_bic.csv",
        "ql_path": OUTPUT_DIR / "concrete_ql_alpha_paths.csv",
        "qln_path": OUTPUT_DIR / "concrete_qln_endpoint_bic_lambda_paths.csv",
        "summary": OUTPUT_DIR / "slide_ready_concrete_endpoint_bic_summary.md",
        "config": OUTPUT_DIR / "config.json",
    }
    interval_frame.to_csv(paths["interval"], index=False)
    point_frame.to_csv(paths["point"], index=False)
    selected_frame.to_csv(paths["selected"], index=False)
    ql_path_frame.to_csv(paths["ql_path"], index=False)
    qln_path_frame.to_csv(paths["qln_path"], index=False)
    paths["summary"].write_text(
        "\n\n".join(
            [
                "# Concrete: Original Endpoint-Wise BIC QLN",
                "## PI Coverage and Mean Width\n"
                + markdown_table(
                    interval_frame[
                        [
                            "Interval",
                            "Model",
                            "Target",
                            "Covered Count",
                            "Test N",
                            "PI Coverage",
                            "Coverage Error",
                            "Mean PI Length",
                            "Median PI Length",
                            "Std PI Length",
                            "Crossing Rate",
                        ]
                    ]
                ),
                "## Selected Parameters\n"
                + markdown_table(
                    selected_frame[
                        [
                            "Model",
                            "Tau",
                            "Selected Parameter",
                            "Selected Count",
                            "Selected Features",
                        ]
                    ]
                ),
                f"## Plot\n- {plot_path.resolve()}",
            ]
        ),
        encoding="utf-8",
    )
    with paths["config"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "Concrete QR/QL/original independent QLN endpoint-wise BIC for 95% and 90% PI",
                "endpoint_taus": list(ENDPOINT_TAUS),
                "intervals": INTERVALS,
                "qln_selection_rule": "endpoint-wise BIC proxy from quantile_lassonet_bic_implementation.py",
                "elapsed_seconds": time.monotonic() - start_time,
                "output_files": {key: str(path.resolve()) for key, path in paths.items() if key != "config"},
                "plot_file": str(plot_path.resolve()),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print("\n=== Concrete PI Coverage / Mean Width ===")
    print(
        interval_frame[
            [
                "Interval",
                "Model",
                "Covered Count",
                "Test N",
                "PI Coverage",
                "Coverage Error",
                "Mean PI Length",
                "Median PI Length",
                "Std PI Length",
                "Crossing Rate",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")
    print(f"Elapsed seconds: {time.monotonic() - start_time:.1f}")


if __name__ == "__main__":
    main()
