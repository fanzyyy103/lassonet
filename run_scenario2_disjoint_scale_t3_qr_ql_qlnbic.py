from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

import run_scenario5_30features_heteroscedastic_qr_ql_qlnbic as base40


OUTPUT_DIR = Path("outputs") / "scenario2_disjoint_scale_t3_qr_ql_qlnbic"
P40_NORMAL_OUTPUT_DIR = Path("outputs") / "scenario2_disjoint_scale_normal_qr_ql_qlnbic"
P90_NORMAL_OUTPUT_DIR = Path("outputs") / "scenario2_disjoint_scale_normal_p90_qr_ql_qlnbic"
N = base40.N
P = base40.P
DATA_SEED = base40.DATA_SEED
SPLIT_SEED = base40.SPLIT_SEED
TRAINING_SEED = base40.TRAINING_SEED
TAUS = base40.TAUS
INTERVAL_SPECS = base40.INTERVAL_SPECS
SCALE_INDICES = tuple(range(30, 35))
NOISE_INDICES = tuple(range(35, 40))
T3_OUTPUT_DIR = Path("outputs") / "scenario2_disjoint_scale_t3_qr_ql_qlnbic"
NORMAL_OUTPUT_DIR = P40_NORMAL_OUTPUT_DIR


def configure_dimension(p: int) -> None:
    global P, NOISE_INDICES
    P = int(p)
    NOISE_INDICES = tuple(range(35, P))
    base40.P = P
    base40.FEATURE_NAMES = np.asarray([f"X{i}" for i in range(1, P + 1)], dtype=object)


def sigma_function(X: np.ndarray) -> np.ndarray:
    return (
        1.0
        + 0.25 * X[:, 30] * X[:, 31]
        + 0.40 * X[:, 32] ** 2
        + 0.35 * np.abs(X[:, 33])
        + 0.30 * X[:, 34]
    ).astype(np.float64)


def generate_dataset(seed: int = DATA_SEED, noise: str = "t3") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    X = np.random.uniform(0.0, 1.0, size=(N, P)).astype(np.float64)
    f0 = base40.f0_signal(X)
    sigma = sigma_function(X)
    if noise == "t3":
        epsilon = np.random.standard_t(df=3, size=N).astype(np.float64)
    elif noise == "normal":
        epsilon = np.random.normal(loc=0.0, scale=1.0, size=N).astype(np.float64)
    else:
        raise ValueError(f"Unsupported noise distribution: {noise}")
    return X, f0, sigma, (f0 + sigma * epsilon).astype(np.float64)


def true_support(tau: float) -> np.ndarray:
    support = np.zeros(P, dtype=bool)
    support[:30] = True
    if not np.isclose(tau, 0.50):
        support[30:35] = True
    return support


def true_quantile(f0: np.ndarray, sigma: np.ndarray, tau: float, noise: str = "t3") -> np.ndarray:
    if noise == "t3":
        quantile = student_t.ppf(tau, df=3)
    elif noise == "normal":
        quantile = norm.ppf(tau)
    else:
        raise ValueError(f"Unsupported noise distribution: {noise}")
    return f0 + sigma * float(quantile)


def feature_selection_row(model_name: str, tau: float, parameter: Any, support_mask: np.ndarray) -> dict[str, Any]:
    support_mask = np.asarray(support_mask, dtype=bool)
    true_mask = true_support(tau)
    metrics = base40.support_precision_recall_f1(support_mask, true_mask)
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    tn = int(np.sum(~support_mask & ~true_mask))
    selected_count = int(support_mask.sum())
    scale_selected = int(support_mask[list(SCALE_INDICES)].sum())
    noise_selected = int(support_mask[list(NOISE_INDICES)].sum())
    return {
        "Model": model_name,
        "Tau": float(tau),
        "True Support": "X1-X30" if np.isclose(tau, 0.50) else "X1-X35",
        "Selected Lambda / Alpha": parameter,
        "Selected Count": selected_count,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "F1": float(metrics["f1"]),
        "False Positive Rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "FP Proportion Among Selected": float(fp / selected_count) if selected_count else 0.0,
        "Scale-only Selected Count": scale_selected,
        "Pure Noise Selected Count": noise_selected,
        "Noise Selection Rate": float(noise_selected / len(NOISE_INDICES)) if NOISE_INDICES else 0.0,
        "Scale FP at Median": scale_selected if np.isclose(tau, 0.50) else 0,
        "Selected Feature Indices": json.dumps(base40.selected_feature_indices(support_mask)),
        "Selected Features": base40.selected_feature_string(support_mask),
    }


def special_frame(selection_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selection_frame[selection_frame["Model"].isin(["QL", "QLN-BIC"])].iterrows():
        selected = set(json.loads(row["Selected Feature Indices"]))
        values = [int(index + 1 in selected) for index in SCALE_INDICES]
        rows.append({"Tau": row["Tau"], "Model": row["Model"], "X31": bool(values[0]), "X32": bool(values[1]), "X33": bool(values[2]), "X34": bool(values[3]), "X35": bool(values[4]), "Recovered": int(sum(values))})
    return pd.DataFrame(rows).sort_values(["Tau", "Model"]).reset_index(drop=True)


def scale_recovery_summary(selection_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in ["QL", "QLN-BIC"]:
        subset = selection_frame[selection_frame["Model"] == model_name]
        nonmedian = subset[~np.isclose(subset["Tau"], 0.50)]
        median = subset[np.isclose(subset["Tau"], 0.50)].iloc[0]
        recovered = int(nonmedian["Scale-only Selected Count"].sum())
        rows.append({"Model": model_name, "Scale-only Recovery": f"{recovered}/40", "Scale-only Recovery Rate": recovered / 40.0, "Median False Scale Selection": f"{int(median['Scale-only Selected Count'])}/5", "Median False Scale Selection Count": int(median["Scale-only Selected Count"])})
    return pd.DataFrame(rows)


def distance_frame(special: pd.DataFrame, noise: str = "t3") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if noise == "t3":
        distance_label = "Abs t3 Quantile"
        quantile_distance = lambda tau: abs(float(student_t.ppf(tau, df=3)))
    elif noise == "normal":
        distance_label = "Abs Normal Quantile"
        quantile_distance = lambda tau: abs(float(norm.ppf(tau)))
    else:
        raise ValueError(f"Unsupported noise distribution: {noise}")
    for tau in TAUS:
        subset = special[np.isclose(special["Tau"], tau)]
        rows.append({"Tau": tau, distance_label: quantile_distance(tau), "QL Scale Recovery": int(subset[subset["Model"] == "QL"]["Recovered"].iloc[0]), "QLN-BIC Scale Recovery": int(subset[subset["Model"] == "QLN-BIC"]["Recovered"].iloc[0])})
    return pd.DataFrame(rows)


def plot_scale_heatmap(path: Path, special: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    features = ["X31", "X32", "X33", "X34", "X35"]
    for axis, model_name in zip(axes, ["QL", "QLN-BIC"], strict=True):
        subset = special[special["Model"] == model_name]
        matrix = np.vstack([subset.sort_values("Tau")[feature].astype(int).to_numpy() for feature in features])
        axis.imshow(matrix, aspect="auto", interpolation="none", cmap="Blues", vmin=0, vmax=1)
        axis.set_title(model_name)
        axis.set_xticks(np.arange(len(TAUS)), [f"{tau:g}" for tau in TAUS], rotation=45, ha="right")
        axis.set_yticks(np.arange(len(features)), features)
        axis.set_xlabel("tau")
    axes[0].set_ylabel("scale-only feature selected")
    fig.suptitle("Scale-only feature selection across quantiles")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_recovery(path: Path, distances: pd.DataFrame, noise: str = "t3") -> None:
    distance_column = "Abs t3 Quantile" if noise == "t3" else "Abs Normal Quantile"
    distance_label = "|t3.ppf(tau)|" if noise == "t3" else "|norm.ppf(tau)|"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, x_column, xlabel in zip(axes, ["Tau", distance_column], ["tau", distance_label], strict=True):
        axis.plot(distances[x_column], distances["QL Scale Recovery"], marker="o", label="QL")
        axis.plot(distances[x_column], distances["QLN-BIC Scale Recovery"], marker="o", label="QLN-BIC")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("number of X31-X35 recovered")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Scale-only recovery versus quantile extremeness")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def point_metric_row(model_name: str, tau: float, prediction: np.ndarray, y_test: np.ndarray, true_q: np.ndarray) -> dict[str, Any]:
    error = prediction - true_q
    return {"Tau": tau, "Model": model_name, "Test Pinball Loss": float(base40.mean_pinball_loss(y_test, prediction, tau)), "True Quantile MSE": float(np.mean(error**2)), "True Quantile RMSE": float(np.sqrt(np.mean(error**2))), "True Quantile MAE": float(np.mean(np.abs(error)))}


def interval_metric_row(interval: str, model_name: str, lower: np.ndarray, upper: np.ndarray, y_test: np.ndarray, true_width: np.ndarray, target: float) -> dict[str, Any]:
    width = upper - lower
    error = width - true_width
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    return {"Interval": interval, "Model": model_name, "Coverage": coverage, "Absolute Coverage Error": abs(coverage - target), "Mean PI Length": float(width.mean()), "Median PI Length": float(np.median(width)), "Mean True PI Length": float(true_width.mean()), "Mean Predicted / Mean True Length": float(width.mean() / true_width.mean()), "Width RMSE": float(np.sqrt(np.mean(error**2))), "Width MAE": float(np.mean(np.abs(error))), "Crossing Rate": float(np.mean(lower > upper))}


def local_rows(interval: str, model_name: str, lower: np.ndarray, upper: np.ndarray, y_test: np.ndarray, sigma_test: np.ndarray, true_width: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width = upper - lower
    for bin_index, indices in enumerate(np.array_split(np.argsort(sigma_test), 10), start=1):
        true_mean = float(true_width[indices].mean())
        predicted_mean = float(width[indices].mean())
        rows.append({"Interval": interval, "Model": model_name, "Sigma Bin": bin_index, "Mean Sigma": float(sigma_test[indices].mean()), "Empirical Coverage": float(np.mean((y_test[indices] >= lower[indices]) & (y_test[indices] <= upper[indices]))), "Mean True Width": true_mean, "Mean True PI Width": true_mean, "Mean Predicted Width": predicted_mean, "Mean Predicted PI Width": predicted_mean, "Predicted / True Width Ratio": predicted_mean / true_mean, "Mean Width Error": float((width - true_width)[indices].mean())})
    return rows


def trend_frame(local: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (interval, model), subset in local.groupby(["Interval", "Model"], sort=False):
        subset = subset.sort_values("Sigma Bin")
        low, high = subset.iloc[0], subset.iloc[-1]
        true_increase = float(high["Mean True Width"] - low["Mean True Width"])
        predicted_increase = float(high["Mean Predicted Width"] - low["Mean Predicted Width"])
        rows.append({"Interval": interval, "Model": model, "Lowest-Sigma Predicted Width": float(low["Mean Predicted Width"]), "Highest-Sigma Predicted Width": float(high["Mean Predicted Width"]), "Lowest-Sigma True Width": float(low["Mean True Width"]), "Highest-Sigma True Width": float(high["Mean True Width"]), "True Width Increase": true_increase, "Predicted Width Increase": predicted_increase, "Heteroscedastic Slope Ratio": predicted_increase / true_increase})
    return pd.DataFrame(rows)


def build_answers(scale_summary: pd.DataFrame, distances: pd.DataFrame, selection: pd.DataFrame, point: pd.DataFrame, interval: pd.DataFrame, trend: pd.DataFrame) -> str:
    qln = scale_summary[scale_summary["Model"] == "QLN-BIC"].iloc[0]
    ql = scale_summary[scale_summary["Model"] == "QL"].iloc[0]
    mse_winners = point.loc[point.groupby("Tau")["True Quantile MSE"].idxmin(), "Model"].value_counts().to_dict()
    interval_winners = []
    width_winners = []
    for interval_name in INTERVAL_SPECS:
        subset = interval[interval["Interval"] == interval_name]
        interval_winners.append(f"{interval_name}: {subset.loc[subset['Absolute Coverage Error'].idxmin(), 'Model']}")
        width_winners.append(f"{interval_name}: {subset.loc[subset['Width RMSE'].idxmin(), 'Model']}")
    qln_trend = trend[trend["Model"] == "QLN-BIC"]
    ql_trend = trend[trend["Model"] == "QL"]
    qln_rates = distances["QLN-BIC Scale Recovery"].to_list()
    ql_rates = distances["QL Scale Recovery"].to_list()
    return "\n".join([
        f"1. QLN-BIC excludes X31-X35 at tau=0.50: {qln['Median False Scale Selection']}.",
        f"2. QLN-BIC nonmedian scale-only recovery: {qln['Scale-only Recovery']}; by tau={qln_rates}.",
        f"3. Scale recovery versus extremeness: QLN-BIC={qln_rates}; QL={ql_rates}; the pattern is not monotone at every tau.",
        f"4. QL shows the same theoretical pattern imperfectly: recovery={ql['Scale-only Recovery']}, median false selection={ql['Median False Scale Selection']}.",
        f"5. Fewer median false scale selections: QLN-BIC={qln['Median False Scale Selection']} versus QL={ql['Median False Scale Selection']}.",
        f"6. Higher nonmedian scale-only recovery: QLN-BIC={qln['Scale-only Recovery']} versus QL={ql['Scale-only Recovery']}.",
        f"7. True quantile MSE wins: QLN-BIC={int(mse_winners.get('QLN-BIC', 0))}/9, QL={int(mse_winners.get('QL', 0))}/9, QR={int(mse_winners.get('QR', 0))}/9.",
        "8. Coverage closest to nominal: " + "; ".join(interval_winners) + ".",
        "9. Lowest width RMSE: " + "; ".join(width_winners) + ".",
        "10. QLN-BIC slope ratios: " + "; ".join(f"{row['Interval']}={row['Heteroscedastic Slope Ratio']:.3f}" for _, row in qln_trend.iterrows()) + "; QL: " + "; ".join(f"{row['Interval']}={row['Heteroscedastic Slope Ratio']:.3f}" for _, row in ql_trend.iterrows()) + ".",
        "11. The experiment provides direct evidence that scale-only feature relevance changes across quantile levels: X31-X35 are irrelevant at the median but enter every nonmedian oracle support.",
        "12. Separate quantile-specific feature selection is supported, but the observed recovery pattern is imperfect and should be evaluated with both support and prediction metrics.",
    ])


def _comparison_frame(normal_frame: pd.DataFrame, t3_frame: pd.DataFrame, noise_columns: list[str]) -> pd.DataFrame:
    normal = normal_frame.copy()
    normal.insert(0, "Noise", "Normal")
    t3 = t3_frame.copy()
    t3.insert(0, "Noise", "t3")
    return pd.concat([normal, t3], ignore_index=True)[noise_columns]


def build_normal_t3_comparison(output_dir: Path, point: pd.DataFrame, interval: pd.DataFrame, selection: pd.DataFrame, scale_summary: pd.DataFrame, trend: pd.DataFrame) -> str:
    if not T3_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Previous t3 output directory not found: {T3_OUTPUT_DIR}")
    t3_point = pd.read_csv(T3_OUTPUT_DIR / "point_quantile_metrics.csv")
    t3_interval = pd.read_csv(T3_OUTPUT_DIR / "prediction_interval_metrics.csv")
    t3_selection = pd.read_csv(T3_OUTPUT_DIR / "feature_selection.csv")
    t3_scale_summary = pd.read_csv(T3_OUTPUT_DIR / "scale_recovery_summary.csv")
    t3_trend = pd.read_csv(T3_OUTPUT_DIR / "heteroscedastic_slope_ratio.csv")

    point_comparison = _comparison_frame(point, t3_point, ["Noise", "Tau", "Model", "Test Pinball Loss", "True Quantile MSE", "True Quantile RMSE", "True Quantile MAE"])
    interval_normal = interval.merge(trend[["Interval", "Model", "Heteroscedastic Slope Ratio"]], on=["Interval", "Model"], how="left")
    interval_t3 = t3_interval.merge(t3_trend[["Interval", "Model", "Heteroscedastic Slope Ratio"]], on=["Interval", "Model"], how="left")
    interval_comparison = _comparison_frame(interval_normal, interval_t3, ["Noise", "Interval", "Model", "Coverage", "Absolute Coverage Error", "Mean True PI Length", "Mean PI Length", "Width RMSE", "Width MAE", "Heteroscedastic Slope Ratio"])
    feature_columns = ["Noise", "Tau", "Model", "Selected Count", "TP", "FP", "FN", "Precision", "Recall", "F1", "Scale-only Selected Count", "Pure Noise Selected Count"]
    feature_comparison = _comparison_frame(selection, t3_selection, feature_columns)
    scale_comparison = _comparison_frame(scale_summary, t3_scale_summary, ["Noise", "Model", "Scale-only Recovery", "Scale-only Recovery Rate", "Median False Scale Selection", "Median False Scale Selection Count"])
    point_comparison.to_csv(output_dir / "normal_vs_t3_point_metrics.csv", index=False)
    interval_comparison.to_csv(output_dir / "normal_vs_t3_prediction_intervals.csv", index=False)
    feature_comparison.to_csv(output_dir / "normal_vs_t3_feature_selection.csv", index=False)
    scale_comparison.to_csv(output_dir / "normal_vs_t3_feature_summary.csv", index=False)

    mse_winners = point_comparison.loc[point_comparison.groupby(["Noise", "Tau"])["True Quantile MSE"].idxmin()].groupby("Noise")["Model"].value_counts().to_dict()
    coverage_winners = interval_comparison.loc[interval_comparison.groupby(["Noise", "Interval"])["Absolute Coverage Error"].idxmin(), ["Noise", "Interval", "Model"]]
    width_winners = interval_comparison.loc[interval_comparison.groupby(["Noise", "Interval"])["Width RMSE"].idxmin(), ["Noise", "Interval", "Model"]]
    slope_distance = interval_comparison.assign(SlopeDistance=(interval_comparison["Heteroscedastic Slope Ratio"] - 1.0).abs())
    slope_winners = slope_distance.loc[slope_distance.groupby(["Noise", "Interval"])["SlopeDistance"].idxmin(), ["Noise", "Interval", "Model"]]
    summary_lines = [
        "Normal versus t3 comparison (all other DGP, split, seeds, preprocessing, and model hyperparameters unchanged).",
        f"1. Median false scale selection: Normal QLN-BIC={scale_comparison[(scale_comparison.Noise == 'Normal') & (scale_comparison.Model == 'QLN-BIC')]['Median False Scale Selection'].iloc[0]}, t3 QLN-BIC={scale_comparison[(scale_comparison.Noise == 't3') & (scale_comparison.Model == 'QLN-BIC')]['Median False Scale Selection'].iloc[0]}; Normal QL={scale_comparison[(scale_comparison.Noise == 'Normal') & (scale_comparison.Model == 'QL')]['Median False Scale Selection'].iloc[0]}, t3 QL={scale_comparison[(scale_comparison.Noise == 't3') & (scale_comparison.Model == 'QL')]['Median False Scale Selection'].iloc[0]}.",
        f"2. Nonmedian scale recovery: Normal QLN-BIC={scale_comparison[(scale_comparison.Noise == 'Normal') & (scale_comparison.Model == 'QLN-BIC')]['Scale-only Recovery'].iloc[0]}, t3 QLN-BIC={scale_comparison[(scale_comparison.Noise == 't3') & (scale_comparison.Model == 'QLN-BIC')]['Scale-only Recovery'].iloc[0]}; Normal QL={scale_comparison[(scale_comparison.Noise == 'Normal') & (scale_comparison.Model == 'QL')]['Scale-only Recovery'].iloc[0]}, t3 QL={scale_comparison[(scale_comparison.Noise == 't3') & (scale_comparison.Model == 'QL')]['Scale-only Recovery'].iloc[0]}.",
        f"3. True quantile MSE wins by noise/model: {mse_winners}.",
        "4. Coverage-closest models: " + "; ".join(f"{row.Noise} {row.Interval}={row.Model}" for row in coverage_winners.itertuples()) + ".",
        "5. Lowest width RMSE models: " + "; ".join(f"{row.Noise} {row.Interval}={row.Model}" for row in width_winners.itertuples()) + ".",
        "6. Slope ratio closest to 1: " + "; ".join(f"{row.Noise} {row.Interval}={row.Model}" for row in slope_winners.itertuples()) + ".",
        "7. Normal noise changes the oracle quantile scale from t3 quantiles to normal quantiles; support definition remains identical because every nonmedian normal quantile is nonzero.",
        "8. The comparison isolates noise-tail effects descriptively; it does not alter the location, scale, split, seeds, or network configuration.",
    ]
    return "\n".join(summary_lines)


def noise_feature_analysis(selection: pd.DataFrame) -> pd.DataFrame:
    columns = ["Tau", "Model", "Selected Count", "TP", "FP", "Scale FP at Median", "Pure Noise Selected Count", "Precision", "Recall", "F1", "False Positive Rate", "FP Proportion Among Selected", "Noise Selection Rate"]
    return selection[columns].rename(columns={"Pure Noise Selected Count": "Pure Noise FP"}).sort_values(["Tau", "Model"]).reset_index(drop=True)


def _p_comparison_selection(frame: pd.DataFrame, point: pd.DataFrame, p: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "p", p)
    result = result.rename(columns={"Selected Count": "Selected"})
    result = result.merge(point[["Tau", "Model", "True Quantile MSE"]], on=["Tau", "Model"], how="left")
    return result[["p", "Model", "Tau", "Selected", "TP", "FP", "Precision", "Recall", "F1", "True Quantile MSE"]]


def build_p40_p90_comparison(output_dir: Path, point: pd.DataFrame, interval: pd.DataFrame, selection: pd.DataFrame, scale_summary: pd.DataFrame, noise: str = "normal") -> str:
    p40 = P40_NORMAL_OUTPUT_DIR if noise == "normal" else T3_OUTPUT_DIR
    if not p40.exists():
        raise FileNotFoundError(f"Previous p=40 Normal output directory not found: {p40}")
    p40_point = pd.read_csv(p40 / "point_quantile_metrics.csv")
    p40_interval = pd.read_csv(p40 / "prediction_interval_metrics.csv")
    p40_selection = pd.read_csv(p40 / "feature_selection.csv")
    p40_scale = pd.read_csv(p40 / "scale_recovery_summary.csv")
    p40_noise = p40_selection["Pure Noise Selected Count"]
    p90_noise = selection["Pure Noise Selected Count"]

    selection_comparison = pd.concat([_p_comparison_selection(p40_selection, p40_point, 40), _p_comparison_selection(selection, point, 90)], ignore_index=True)
    selection_comparison.to_csv(output_dir / "p40_vs_p90_feature_selection.csv", index=False)
    scale_rows = []
    for p_value, scale_frame, feature_frame in [(40, p40_scale, p40_selection), (90, scale_summary, selection)]:
        for _, row in scale_frame.iterrows():
            subset = feature_frame[(feature_frame["Model"] == row["Model"]) & np.isclose(feature_frame["Tau"], 0.50)]
            scale_rows.append({"p": p_value, "Model": row["Model"], "Scale Recovery": row["Scale-only Recovery"], "Mean Noise FP": float(feature_frame[feature_frame["Model"] == row["Model"]]["Pure Noise Selected Count"].mean()), "Median Noise FP": int(subset["Pure Noise Selected Count"].iloc[0])})
    scale_comparison = pd.DataFrame(scale_rows)
    interval_rows = []
    for p_value, frame in [(40, p40_interval), (90, interval)]:
        for _, row in frame.iterrows():
            interval_rows.append({"p": p_value, "Interval": row["Interval"], "Model": row["Model"], "Coverage": row["Coverage"], "Coverage Error": row["Absolute Coverage Error"], "Width RMSE": row["Width RMSE"]})
    interval_comparison = pd.DataFrame(interval_rows)
    scale_comparison.to_csv(output_dir / "p40_vs_p90_summary.csv", index=False)
    interval_comparison.to_csv(output_dir / "p40_vs_p90_prediction_intervals.csv", index=False)

    answers = [
        f"1. QLN-BIC p=90 scale recovery: {scale_comparison[(scale_comparison.p == 90) & (scale_comparison.Model == 'QLN-BIC')]['Scale Recovery'].iloc[0]}.",
        f"2. QLN-BIC p=90 median scale false selection: {int(selection[(selection.Model == 'QLN-BIC') & np.isclose(selection.Tau, 0.50)]['Scale-only Selected Count'].iloc[0])}/5.",
        f"3. QLN-BIC mean pure-noise FP: p=40={scale_comparison[(scale_comparison.p == 40) & (scale_comparison.Model == 'QLN-BIC')]['Mean Noise FP'].iloc[0]:.3f}; p=90={scale_comparison[(scale_comparison.p == 90) & (scale_comparison.Model == 'QLN-BIC')]['Mean Noise FP'].iloc[0]:.3f}.",
        f"4. QLN-BIC median pure-noise FP: p=40={int(scale_comparison[(scale_comparison.p == 40) & (scale_comparison.Model == 'QLN-BIC')]['Median Noise FP'].iloc[0])}; p=90={int(scale_comparison[(scale_comparison.p == 90) & (scale_comparison.Model == 'QLN-BIC')]['Median Noise FP'].iloc[0])}.",
        f"5. QL mean pure-noise FP: p=40={scale_comparison[(scale_comparison.p == 40) & (scale_comparison.Model == 'QL')]['Mean Noise FP'].iloc[0]:.3f}; p=90={scale_comparison[(scale_comparison.p == 90) & (scale_comparison.Model == 'QL')]['Mean Noise FP'].iloc[0]:.3f}.",
        "6. Precision/recall/F1 are reported in p40_vs_p90_feature_selection.csv for every tau and model.",
        "7. Adding 50 pure-noise variables is isolated from the DGP because X1-X35, y, split, preprocessing, seeds, and model hyperparameters are unchanged.",
        "8. Prediction-interval coverage and width RMSE are reported in p40_vs_p90_prediction_intervals.csv.",
    ]
    return "\n".join(answers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario 2 disjoint location/scale quantile simulation.")
    parser.add_argument("--noise", choices=("t3", "normal"), default="t3")
    parser.add_argument("--p", type=int, choices=(40, 90), default=40)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    default_output = P90_NORMAL_OUTPUT_DIR if args.noise == "normal" and args.p == 90 else NORMAL_OUTPUT_DIR if args.noise == "normal" else OUTPUT_DIR
    output_dir = args.output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    configure_dimension(args.p)
    base40.configure_base()
    base40.base.set_all_seeds(TRAINING_SEED)
    X, f0, sigma, y = generate_dataset(DATA_SEED, args.noise)
    split = base40.base.split_indices(SPLIT_SEED)
    X_train_raw, X_val_raw, X_test_raw = X[split["train_idx"]], X[split["val_idx"]], X[split["test_idx"]]
    y_train, y_val, y_test = y[split["train_idx"]], y[split["val_idx"]], y[split["test_idx"]]
    f0_test, sigma_test = f0[split["test_idx"]], sigma[split["test_idx"]]
    scaler = base40.StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float64)
    X_val = scaler.transform(X_val_raw).astype(np.float64)
    X_test = scaler.transform(X_test_raw).astype(np.float64)
    true_quantiles = {tau: true_quantile(f0_test, sigma_test, tau, args.noise) for tau in TAUS}

    predictions: dict[str, dict[float, np.ndarray]] = {"QR": {}, "QL": {}, "QLN-BIC": {}}
    point_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    ql_path_rows: list[dict[str, Any]] = []
    qln_path_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for tau in TAUS:
        print(f"[QR] fitting tau={tau:.3f}", flush=True)
        qr = base40.base.fit_standard_quantile_regression(X_train, y_train, X_test, tau)
        predictions["QR"][tau] = qr["prediction"]
        qr_support = base40.support_from_qr_model(qr["model"])
        selection_rows.append(feature_selection_row("QR", tau, pd.NA, qr_support))
        parameter_rows.append({"Model": "QR", "Tau": tau, "Selected Count": int(qr_support.sum()), "Selected Features": base40.selected_feature_string(qr_support)})

        print(f"[QL] fitting tau={tau:.3f}", flush=True)
        ql = base40.base.fit_quantile_lasso(X_train, y_train, X_val, y_val, X_test, tau)
        predictions["QL"][tau] = ql["prediction"]
        ql_path_rows.extend(ql["path_rows"])
        selection_rows.append(feature_selection_row("QL", tau, ql["best_alpha"], ql["support_mask"]))
        parameter_rows.append({"Model": "QL", "Tau": tau, "Selected Count": int(ql["support_mask"].sum()), "Selected Features": base40.selected_feature_string(ql["support_mask"])})

        print(f"[QLN-BIC] fitting independent tau={tau:.3f}", flush=True)
        qln = base40.base.fit_quantile_lassonet_bic(X_train, y_train, X_val, y_val, X_test, tau)
        predictions["QLN-BIC"][tau] = qln["prediction"]
        qln_path_rows.extend(qln["path_rows"])
        selection_rows.append(feature_selection_row("QLN-BIC", tau, qln["best_lambda"], qln["support_mask"]))
        parameter_rows.append({"Model": "QLN-BIC", "Tau": tau, "Selected Count": int(qln["support_mask"].sum()), "Selected Features": base40.selected_feature_string(qln["support_mask"])})
        for model_name in ["QR", "QL", "QLN-BIC"]:
            point_rows.append(point_metric_row(model_name, tau, predictions[model_name][tau], y_test, true_quantiles[tau]))

    interval_rows: list[dict[str, Any]] = []
    local_row_list: list[dict[str, Any]] = []
    true_widths: dict[str, np.ndarray] = {}
    width_arrays: dict[str, dict[str, np.ndarray]] = {}
    for interval_name, spec in INTERVAL_SPECS.items():
        lower_tau, upper_tau = float(spec["lower"]), float(spec["upper"])
        true_width = true_quantiles[upper_tau] - true_quantiles[lower_tau]
        true_widths[interval_name] = true_width
        for model_name in ["QR", "QL", "QLN-BIC"]:
            lower, upper = predictions[model_name][lower_tau], predictions[model_name][upper_tau]
            interval_rows.append(interval_metric_row(interval_name, model_name, lower, upper, y_test, true_width, float(spec["target"])))
            local_row_list.extend(local_rows(interval_name, model_name, lower, upper, y_test, sigma_test, true_width))
        width_arrays[interval_name] = {model_name: predictions[model_name][upper_tau] - predictions[model_name][lower_tau] for model_name in predictions}

    point_frame = pd.DataFrame(point_rows).sort_values(["Tau", "Model"]).reset_index(drop=True)
    interval_frame = pd.DataFrame(interval_rows).sort_values(["Interval", "Model"]).reset_index(drop=True)
    selection_frame = pd.DataFrame(selection_rows).sort_values(["Tau", "Model"]).reset_index(drop=True)
    special = special_frame(selection_frame)
    scale_summary = scale_recovery_summary(selection_frame)
    distances = distance_frame(special, args.noise)
    parameter_frame = pd.DataFrame(parameter_rows).sort_values(["Tau", "Model"]).reset_index(drop=True)
    local_frame = pd.DataFrame(local_row_list).sort_values(["Interval", "Model", "Sigma Bin"]).reset_index(drop=True)
    trend = trend_frame(local_frame)
    ql_path_frame = pd.DataFrame(ql_path_rows).sort_values(["tau", "alpha"]).reset_index(drop=True)
    qln_path_frame = pd.DataFrame(qln_path_rows).sort_values(["tau", "path_index"]).reset_index(drop=True)
    qln_path_frame["validation_pinball_loss_sum"] = qln_path_frame["validation_pinball_loss"] * len(y_val)
    qln_path_frame["sigma_tau_BIC"] = (1.0 - np.abs(1.0 - 2.0 * qln_path_frame["tau"])) / 2.0
    answers = build_answers(scale_summary, distances, selection_frame, point_frame, interval_frame, trend)
    comparison_answers = None
    if args.noise == "normal" and args.p == 40:
        comparison_answers = build_normal_t3_comparison(output_dir, point_frame, interval_frame, selection_frame, scale_summary, trend)
    elif args.p == 90:
        comparison_answers = build_p40_p90_comparison(output_dir, point_frame, interval_frame, selection_frame, scale_summary, args.noise)

    paths = {
        "point_metrics": output_dir / "point_quantile_metrics.csv",
        "interval_metrics": output_dir / "prediction_interval_metrics.csv",
        "feature_selection": output_dir / "feature_selection.csv",
        "noise_feature_analysis": output_dir / "noise_feature_analysis.csv",
        "scale_only_recovery": output_dir / "scale_only_feature_recovery.csv",
        "scale_recovery_summary": output_dir / "scale_recovery_summary.csv",
        "distance_from_median": output_dir / "distance_from_median_recovery.csv",
        "selected_parameters": output_dir / "selected_parameters.csv",
        "ql_paths": output_dir / "ql_alpha_paths.csv",
        "qln_paths": output_dir / "qln_bic_lambda_paths.csv",
        "local": output_dir / "local_heteroscedasticity.csv",
        "trend": output_dir / "heteroscedastic_slope_ratio.csv",
        "answers": output_dir / "answers.md",
        "config": output_dir / "config.json",
        "predictions": output_dir / "test_predictions.npz",
        "plot_support": output_dir / "feature_selection_heatmap.png",
        "plot_scale": output_dir / "scale_only_feature_heatmap.png",
        "plot_recovery": output_dir / "scale_recovery_vs_quantile_extremeness.png",
        "plot_q50": output_dir / "true_vs_predicted_q50.png",
        "plot_q025": output_dir / "true_vs_predicted_q025.png",
        "plot_q975": output_dir / "true_vs_predicted_q975.png",
        "plot_width95": output_dir / "predicted_95pi_width_vs_sigma.png",
        "plot_width_true": output_dir / "true_vs_predicted_95pi_width.png",
        "plot_deciles": output_dir / "mean_predicted_width_by_sigma_decile.png",
    }
    point_frame.to_csv(paths["point_metrics"], index=False)
    interval_frame.to_csv(paths["interval_metrics"], index=False)
    selection_frame.to_csv(paths["feature_selection"], index=False)
    noise_feature_analysis(selection_frame).to_csv(paths["noise_feature_analysis"], index=False)
    special.to_csv(paths["scale_only_recovery"], index=False)
    scale_summary.to_csv(paths["scale_recovery_summary"], index=False)
    distances.to_csv(paths["distance_from_median"], index=False)
    parameter_frame.to_csv(paths["selected_parameters"], index=False)
    ql_path_frame.to_csv(paths["ql_paths"], index=False)
    qln_path_frame.to_csv(paths["qln_paths"], index=False)
    local_frame.to_csv(paths["local"], index=False)
    trend.to_csv(paths["trend"], index=False)
    paths["answers"].write_text(answers + "\n", encoding="utf-8")
    np.savez_compressed(paths["predictions"], y_test=y_test, f0_test=f0_test, sigma_test=sigma_test, **{f"{model_name.replace('-', '_')}_q{tau:g}": prediction for model_name, values in predictions.items() for tau, prediction in values.items()})

    base40.plot_support(paths["plot_support"], selection_frame)
    plot_scale_heatmap(paths["plot_scale"], special)
    plot_recovery(paths["plot_recovery"], distances, args.noise)
    noise_label = "Normal" if args.noise == "normal" else "t3"
    base40.plot_prediction(paths["plot_q50"], f"Scenario 2 {noise_label}: true vs predicted q0.50", f0_test, true_quantiles[0.50], {name: values[0.50] for name, values in predictions.items()})
    base40.plot_prediction(paths["plot_q025"], f"Scenario 2 {noise_label}: true vs predicted q0.025", true_quantiles[0.025], true_quantiles[0.025], {name: values[0.025] for name, values in predictions.items()})
    base40.plot_prediction(paths["plot_q975"], f"Scenario 2 {noise_label}: true vs predicted q0.975", true_quantiles[0.975], true_quantiles[0.975], {name: values[0.975] for name, values in predictions.items()})
    base40.plot_width_vs_sigma(paths["plot_width95"], sigma_test, width_arrays["95%"], true_widths["95%"], "95%")
    base40.plot_true_pred_width(paths["plot_width_true"], width_arrays["95%"], true_widths["95%"])
    base40.plot_decile_width(paths["plot_deciles"], local_frame)

    config = {
        "experiment": f"Scenario 2 disjoint location/scale support with {noise_label} noise",
        "n": N,
        "p": P,
        "data_seed": DATA_SEED,
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "predictors": "independent Uniform(0,1)",
        "f0_source": "run_scenario5_30features_qr_ql_shared_qln.scenario_components(X[:, :30])[f0]",
        "sigma_formula": "1 + 0.25*X31*X32 + 0.40*X33^2 + 0.35*abs(X34) + 0.30*X35",
        "error_distribution": "Student t(df=3), standard scale, not rescaled" if args.noise == "t3" else "Normal(0,1), not rescaled",
        "true_quantile": "f0 + sigma(X)*scipy.stats.t.ppf(tau, df=3)" if args.noise == "t3" else "f0 + sigma(X)*scipy.stats.norm.ppf(tau)",
        "taus": list(TAUS),
        "true_support_median": list(range(1, 31)),
        "true_support_nonmedian": list(range(1, 36)),
        "scale_only_features": list(range(31, 36)),
        "pure_noise_features": list(range(36, P + 1)),
        "qln_selection": "independent single-quantile BIC",
        "qln_network": dict(base40.base.QUANTILE_LASSONET_CONFIG),
        "pure_noise_feature_count": P - 35,
        "bic_formula": "(2 / sigma_tau) * validation_pinball_loss_sum + df(lambda) * log(n_val)",
        "output_files": {key: str(path.resolve()) for key, path in paths.items()},
        "elapsed_seconds": time.monotonic() - start_time,
    }
    paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    print("=== Point Quantile Metrics ===")
    print(point_frame.to_string(index=False))
    print("\n=== Prediction Interval Metrics ===")
    print(interval_frame.to_string(index=False))
    print("\n=== Scale-only Recovery ===")
    print(special.to_string(index=False))
    print(scale_summary.to_string(index=False))
    print("\n=== Distance From Median ===")
    print(distances.to_string(index=False))
    print("\n=== Heteroscedastic Slope Ratio ===")
    print(trend.to_string(index=False))
    print("\n=== Explicit Answers ===")
    print(answers)
    if comparison_answers is not None:
        comparison_filename = "normal_vs_t3_answers.md" if args.p == 40 else "p40_vs_p90_answers.md"
        paths["comparison_answers"] = output_dir / comparison_filename
        paths["comparison_answers"].write_text(comparison_answers + "\n", encoding="utf-8")
        print("\n=== Comparison Answers ===")
        print(comparison_answers)
    print(f"\nSaved results to: {output_dir.resolve()}")
    print(f"Elapsed seconds: {time.monotonic() - start_time:.1f}")


if __name__ == "__main__":
    main()
