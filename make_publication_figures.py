# -*- coding: utf-8 -*-
"""Generate publication-style figures and evidence tables for the report.

The script reads the saved real-data experiment outputs under
``code/runs/ngsim_8000_tracks`` by default. It does not rerun model training, so
it is fast and deterministic. Use ``run_experiments_gpu_more.py`` first when the
raw experiment results need to be regenerated on GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle


PALETTE = {
    "CV": "#7A8793",
    "CA": "#9AA4AE",
    "Ridge": "#4B88A2",
    "Ridge-nonlinear-only": "#9BC8C9",
    "Ridge+nonlinear": "#7FB3B5",
    "XGBoost": "#B07AA1",
    "XGBoost-nonlinear-only": "#C7A4BB",
    "XGBoost+nonlinear": "#D95F02",
    "signal": "#D95F02",
    "blue": "#3B6EA8",
    "teal": "#4C9A8A",
    "grey": "#6B7280",
    "light": "#E8ECEF",
    "dark": "#1F2933",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _embed(series: np.ndarray, m: int = 3, tau: int = 4) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    start = (m - 1) * tau
    if len(series) <= start + 1:
        return np.empty((0, m))
    return np.asarray([[series[i - j * tau] for j in range(m)] for i in range(start, len(series))])


def recurrence(series: np.ndarray, m: int = 3, tau: int = 4, quantile: float = 0.15) -> np.ndarray:
    emb = _embed(series, m=m, tau=tau)
    if len(emb) < 3:
        return np.zeros((1, 1))
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    nz = dist[dist > 0]
    eps = np.quantile(nz, quantile) if len(nz) else 0.0
    return dist <= eps


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < m + 4:
        return np.nan
    r = r_ratio * np.nanstd(x)
    if not np.isfinite(r) or r <= 1e-12:
        return 0.0

    def count(mm: int) -> float:
        wins = np.asarray([x[i : i + mm] for i in range(len(x) - mm + 1)])
        total = 0
        for i in range(len(wins)):
            d = np.max(np.abs(wins[i + 1 :] - wins[i]), axis=1)
            total += int(np.sum(d <= r))
        return float(total)

    b = count(m)
    a = count(m + 1)
    return float(np.log((b + 1.0) / (a + 1.0)))


def dfa_alpha(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 16 or np.nanstd(x) < 1e-12:
        return np.nan
    y = np.cumsum(x - np.nanmean(x))
    sizes = np.asarray([4, 5, 6, 8, 10, 12, 15])
    fs = []
    used = []
    for s in sizes:
        nseg = len(y) // s
        if nseg < 2:
            continue
        vals = []
        t = np.arange(s)
        for k in range(nseg):
            seg = y[k * s : (k + 1) * s]
            p = np.polyfit(t, seg, 1)
            vals.append(np.sqrt(np.mean((seg - np.polyval(p, t)) ** 2)))
        fs.append(np.sqrt(np.mean(np.asarray(vals) ** 2)))
        used.append(s)
    if len(fs) < 2:
        return np.nan
    return float(np.polyfit(np.log(used), np.log(np.asarray(fs) + 1e-12), 1)[0])


def lyapunov_proxy(y: np.ndarray, m: int = 3, tau: int = 4) -> float:
    e = _embed(y, m=m, tau=tau)
    if len(e) < 4:
        return np.nan
    d1 = np.linalg.norm(e[1:] - e[:-1], axis=1) + 1e-8
    d2 = np.linalg.norm(e[2:] - e[:-2], axis=1) + 1e-8
    return float(np.clip(np.mean(np.log(d2 / d1[1:])), -4, 4))


def load_context(run_dir: Path) -> dict[str, object]:
    res = run_dir / "results"
    gen = run_dir / "generated" / "ngsim_compatible_benchmark.csv"
    if not gen.exists():
        gen = run_dir / "data" / "generated" / "ngsim_compatible_benchmark.csv"
    return {
        "metrics": pd.read_csv(res / "prediction_metrics.csv"),
        "horizon": pd.read_csv(res / "horizon_errors.csv"),
        "ablation": pd.read_csv(res / "ablation.csv"),
        "embed": pd.read_csv(res / "sensitivity_embedding.csv"),
        "noise": pd.read_csv(res / "noise_robustness.csv"),
        "cls": pd.read_csv(res / "classification_metrics.csv"),
        "imp": pd.read_csv(res / "feature_importance.csv"),
        "mse": pd.read_csv(res / "mse_by_behavior.csv"),
        "pred": np.load(res / "predictions_test.npz"),
        "meta": json.loads((res / "experiment_metadata.json").read_text(encoding="utf-8")),
        "tracks": pd.read_csv(gen),
    }


def fig00_framework(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.94, 2.75))
    ax.axis("off")
    nodes = [
        (0.02, 0.58, 0.22, 0.20, "3 s history\nvehicle track", PALETTE["light"]),
        (0.31, 0.76, 0.22, 0.15, "CV prior\ninertial motion", "#EEF4FA"),
        (0.31, 0.54, 0.22, 0.15, "phase space\nm, tau embedding", "#EEF8F6"),
        (0.31, 0.32, 0.22, 0.15, "Lyapunov/RQA\nentropy/DFA", "#EEF8F6"),
        (0.59, 0.64, 0.22, 0.15, "local graph\nheadway/density", "#F4F0F6"),
        (0.59, 0.39, 0.22, 0.15, "validated fusion\nalpha selection", "#F7F1E8"),
        (0.84, 0.53, 0.14, 0.18, "XGBoost\nresidual", "#FDEDE5"),
        (0.84, 0.25, 0.14, 0.15, "5 s\nforecast", "#FDEDE5"),
    ]
    for x, y, w, h, text, color in nodes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=PALETTE["dark"], lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.7)
    arrows = [
        ((0.24, 0.68), (0.31, 0.84)),
        ((0.24, 0.68), (0.31, 0.61)),
        ((0.24, 0.68), (0.31, 0.39)),
        ((0.53, 0.84), (0.59, 0.71)),
        ((0.53, 0.61), (0.59, 0.47)),
        ((0.53, 0.39), (0.59, 0.47)),
        ((0.81, 0.47), (0.84, 0.62)),
        ((0.91, 0.53), (0.91, 0.40)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.65, color=PALETTE["dark"]))
    ax.text(0.50, 0.11, r"$\hat{Y}=Y_{\mathrm{CV}}+\alpha f_{\mathrm{NL}}+(1-\alpha)f_{\mathrm{base}}$", ha="center", fontsize=7.2)
    save_figure(fig, fig_dir, "fig00_framework")


def fig01_tracks(fig_dir: Path, tracks: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    for _, g in list(tracks.groupby("Vehicle_ID", sort=True))[:18]:
        ax.plot(g["Local_Y"], g["Local_X"], lw=0.75, alpha=0.72)
    ax.set_xlabel("Longitudinal position y (m)")
    ax.set_ylabel("Lateral position x (m)")
    ax.set_title("Representative cleaned NGSIM tracks")
    ax.grid(True, color="#E5E7EB", lw=0.35)
    save_figure(fig, fig_dir, "fig01_trajectory_samples")


def fig02_phase(fig_dir: Path, tracks: pd.DataFrame) -> None:
    g = next(g for _, g in tracks.groupby("Vehicle_ID", sort=True) if len(g) > 80)
    y = g["Local_Y"].to_numpy()[:80]
    e = _embed(y, m=3, tau=4)
    rec = recurrence(y, m=3, tau=4)
    fig, axes = plt.subplots(1, 2, figsize=(3.94, 1.86))
    panel_label(axes[0], "a")
    axes[0].plot(e[:, 0], e[:, 1], "-o", ms=1.8, lw=0.55, color=PALETTE["blue"])
    axes[0].set_xlabel(r"$y(t)$ (m)")
    axes[0].set_ylabel(r"$y(t-\tau)$ (m)")
    axes[0].set_title("Delay-coordinate projection")
    panel_label(axes[1], "b")
    axes[1].imshow(rec, origin="lower", interpolation="nearest", cmap="Greys", aspect="auto")
    axes[1].set_xlabel("State index")
    axes[1].set_ylabel("State index")
    axes[1].set_title("Recurrence structure")
    save_figure(fig, fig_dir, "fig02_phase_recurrence")


def fig03_features(fig_dir: Path, tracks: pd.DataFrame) -> None:
    rows = []
    for _, g in list(tracks.groupby("Vehicle_ID", sort=True))[:220]:
        h = g.sort_values("Frame_ID").head(30)
        if len(h) < 30:
            continue
        x = h["Local_X"].to_numpy()
        y = h["Local_Y"].to_numpy()
        speed = np.sqrt(np.gradient(x, 0.1) ** 2 + np.gradient(y, 0.1) ** 2)
        rows.append(
            {
                "SampEn(speed)": sample_entropy(speed),
                "DFA alpha": dfa_alpha(speed),
                "Lyapunov proxy": lyapunov_proxy(y),
                "Lateral range": float(np.ptp(x)),
            }
        )
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(3.94, 1.92))
    panel_label(axes[0], "a")
    axes[0].boxplot(
        [df["SampEn(speed)"], df["DFA alpha"], df["Lyapunov proxy"]],
        tick_labels=["SampEn", "DFA", "Lyap."],
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor="#EEF4FA", color=PALETTE["blue"], lw=0.7),
        medianprops=dict(color=PALETTE["signal"], lw=0.9),
        whiskerprops=dict(color=PALETTE["grey"], lw=0.65),
        capprops=dict(color=PALETTE["grey"], lw=0.65),
        flierprops=dict(marker=".", markersize=1.6, markerfacecolor=PALETTE["grey"], markeredgecolor=PALETTE["grey"], alpha=0.45),
    )
    axes[0].set_ylabel("Feature value")
    axes[0].set_title("Finite-window nonlinear descriptors")
    panel_label(axes[1], "b")
    axes[1].scatter(df["Lyapunov proxy"], df["Lateral range"], s=9, alpha=0.68, color=PALETTE["teal"], edgecolor="none")
    axes[1].set_xlabel("Lyapunov proxy")
    axes[1].set_ylabel("Lateral range (m)")
    axes[1].set_title("Dynamic instability vs. lateral motion")
    save_figure(fig, fig_dir, "fig03_nonlinear_feature_distribution")


def fig04_metrics(fig_dir: Path, metrics: pd.DataFrame) -> None:
    order = ["CV", "CA", "Ridge", "Ridge-nonlinear-only", "Ridge+nonlinear", "XGBoost", "XGBoost-nonlinear-only", "XGBoost+nonlinear"]
    m = metrics.set_index("Model").loc[order].reset_index()
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(5.15, 2.65))
    colors = [PALETTE.get(k, PALETTE["grey"]) for k in m["Model"]]
    ax.bar(x - 0.18, m["ADE"], width=0.34, color=colors, alpha=0.92, label="ADE")
    ax.bar(x + 0.18, m["FDE"], width=0.34, color=colors, alpha=0.42, label="FDE")
    ax.set_ylabel("Displacement error (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(["CV", "CA", "Ridge", "Ridge\nNL-only", "Ridge\n+NL", "XGB", "XGB\nNL-only", "XGB\n+NL"], rotation=18, ha="right")
    ax.set_title("Prediction accuracy on held-out samples")
    ax.legend(ncol=2, loc="upper right")
    for i, val in enumerate(m["ADE"]):
        ax.text(i - 0.18, val + 0.06, f"{val:.2f}", ha="center", va="bottom", fontsize=5.7)
    save_figure(fig, fig_dir, "fig04_prediction_metrics")


def fig05_horizon(fig_dir: Path, horizon: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.94, 2.45))
    order = ["CV", "CA", "Ridge", "Ridge+nonlinear", "XGBoost", "XGBoost+nonlinear"]
    for name in order:
        if name not in set(horizon["Model"]):
            continue
        g = horizon[horizon["Model"] == name]
        ax.plot(g["Horizon_s"], g["Error_m"], marker="o", ms=2.6, lw=1.0, label=name, color=PALETTE.get(name, None))
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel("Mean displacement error (m)")
    ax.set_title("Error accumulation over the 5 s forecast")
    ax.grid(True, color="#E5E7EB", lw=0.35)
    ax.legend(ncol=2, loc="upper left")
    save_figure(fig, fig_dir, "fig05_horizon_error")


def fig06_ablation(fig_dir: Path, ablation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.94, 2.45))
    x = np.arange(len(ablation))
    base = float(ablation["ADE"].iloc[0])
    delta = base - ablation["ADE"].to_numpy()
    colors = [PALETTE["grey"] if d <= 0 else PALETTE["teal"] for d in delta]
    ax.axhline(0, color=PALETTE["dark"], lw=0.65)
    ax.bar(x, delta * 100, color=colors, width=0.62)
    ax.set_ylabel("ADE reduction vs. kinematics (cm)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Kin.", "+Delay", "+Lyap.", "+RQA", "+Ent/DFA", "+Graph"], rotation=25, ha="right")
    ax.set_title("Ablation isolates useful nonlinear evidence")
    for i, d in enumerate(delta * 100):
        ax.text(i, d + (0.25 if d >= 0 else -0.5), f"{d:.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=5.7)
    save_figure(fig, fig_dir, "fig06_ablation")


def fig07_embedding(fig_dir: Path, embed: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(embed["Embedding_dim"], embed["ADE"], "-o", ms=2.8, lw=1.0, color=PALETTE["blue"], label="ADE")
    ax.plot(embed["Embedding_dim"], embed["FDE"], "-s", ms=2.6, lw=1.0, color=PALETTE["signal"], label="FDE")
    ax.set_xlabel("Embedding dimension m")
    ax.set_ylabel("Error (m, log scale)")
    ax.set_yscale("log")
    ax.set_title("Embedding dimension sensitivity")
    ax.grid(True, which="both", color="#E5E7EB", lw=0.35)
    ax.legend()
    save_figure(fig, fig_dir, "fig07_embedding_sensitivity")


def fig08_mse(fig_dir: Path, mse: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    behavior_colors = {"Following": PALETTE["blue"], "Lane-change": PALETTE["signal"], "Hard-brake": PALETTE["teal"]}
    for name, g in mse.groupby("Behavior", sort=False):
        ax.plot(g["Scale"], g["SampEn"], "-o", ms=2.7, lw=1.0, color=behavior_colors.get(name), label=name)
    ax.set_xlabel("Coarse-graining scale")
    ax.set_ylabel("Sample entropy")
    ax.set_title("Multiscale complexity by driving behavior")
    ax.grid(True, color="#E5E7EB", lw=0.35)
    ax.legend()
    save_figure(fig, fig_dir, "fig08_multiscale_complexity")


def fig09_noise(fig_dir: Path, noise: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for name, g in noise.groupby("Model", sort=False):
        ax.plot(g["NoisePercent"], g["ADE"], "-o", ms=2.8, lw=1.0, color=PALETTE.get(name), label=name)
    ax.set_xlabel("Observation noise (% of lane width)")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Robustness to observation perturbation")
    ax.grid(True, color="#E5E7EB", lw=0.35)
    ax.legend()
    save_figure(fig, fig_dir, "fig09_noise_robustness")


def fig10_lane(fig_dir: Path, cls: pd.DataFrame, run_dir: Path) -> None:
    row = cls[cls["Model"] == "XGBoost+nonlinear"].iloc[0]
    cm = np.asarray([[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]])
    roc = pd.read_csv(run_dir / "results" / "roc_curve_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.94, 1.92))
    panel_label(axes[0], "a")
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1], ["Stay", "Change"])
    axes[0].set_yticks([0, 1], ["Stay", "Change"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Confusion matrix")
    threshold = cm.max() / 2
    for (i, j), val in np.ndenumerate(cm):
        axes[0].text(j, i, str(val), ha="center", va="center", fontsize=6.8, color="white" if val > threshold else PALETTE["dark"])
    panel_label(axes[1], "b")
    axes[1].plot(roc["fpr"], roc["tpr"], color=PALETTE["signal"], lw=1.1, label=f"AUC={row['AUC']:.3f}")
    axes[1].plot([0, 1], [0, 1], ls="--", lw=0.65, color=PALETTE["grey"])
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curve")
    axes[1].legend(loc="lower right")
    save_figure(fig, fig_dir, "fig10_lane_change")


def fig11_importance(fig_dir: Path, imp: pd.DataFrame) -> None:
    top = imp.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.55, 2.92))
    colors = [PALETTE["teal"] if any(k in f for k in ["lyapunov", "delay", "dfa", "sampen", "mse", "graph"]) else PALETTE["blue"] for f in top["Feature"]]
    ax.barh(top["Feature"], top["Importance"], color=colors)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Model uses history and nonlinear descriptors")
    save_figure(fig, fig_dir, "fig11_feature_importance")


def fig12_cases(fig_dir: Path, pred: np.lib.npyio.NpzFile) -> None:
    y = pred["Yte"].reshape(-1, 50, 2)
    cv = pred["CV"].reshape(-1, 50, 2)
    prop = pred["XGBoost_nonlinear"].reshape(-1, 50, 2)
    xgb = pred["XGBoost"].reshape(-1, 50, 2)
    err = np.abs(y[:, -1, 0] - cv[:, -1, 0]) - np.abs(y[:, -1, 0] - prop[:, -1, 0])
    cases = np.argsort(err)[-3:][::-1]
    fig = plt.figure(figsize=(3.94, 2.65))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    for i, k in enumerate(cases):
        ax = fig.add_subplot(gs[0, i])
        panel_label(ax, chr(ord("a") + i))
        y0 = y[k, 0, :].copy()
        yy = y[k] - y0
        ccv = cv[k] - y0
        xxgb = xgb[k] - y0
        pp = prop[k] - y0
        ax.plot(yy[:, 1], yy[:, 0], lw=1.25, color=PALETTE["dark"], label="Observed")
        ax.plot(ccv[:, 1], ccv[:, 0], lw=0.85, color=PALETTE["CV"], label="CV")
        ax.plot(xxgb[:, 1], xxgb[:, 0], lw=0.9, color=PALETTE["XGBoost"], label="XGBoost")
        ax.plot(pp[:, 1], pp[:, 0], lw=1.05, color=PALETTE["signal"], label="XGB+NL")
        ax.set_xlabel(r"$\Delta y$ (m)")
        if i == 0:
            ax.set_ylabel(r"$\Delta x$ (m)")
        ax.set_title(f"case {i + 1}")
        ax.grid(True, color="#E5E7EB", lw=0.35)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    save_figure(fig, fig_dir, "fig12_qualitative_cases")


def write_evidence_table(table_dir: Path, ctx: dict[str, object]) -> None:
    metrics = ctx["metrics"].set_index("Model")
    noise = ctx["noise"]
    meta = ctx["meta"]

    cv_ade = float(metrics.loc["CV", "ADE"])
    ridge = float(metrics.loc["Ridge", "ADE"])
    ridge_nl = float(metrics.loc["Ridge+nonlinear", "ADE"])
    xgb = float(metrics.loc["XGBoost", "ADE"])
    xgb_nl = float(metrics.loc["XGBoost+nonlinear", "ADE"])
    fde_xgb = float(metrics.loc["XGBoost", "FDE"])
    fde_xgb_nl = float(metrics.loc["XGBoost+nonlinear", "FDE"])
    n10 = noise[noise["NoisePercent"] == 10].set_index("Model")
    robust = float(n10.loc["XGBoost", "ADE"] - n10.loc["XGBoost+nonlinear", "ADE"]) * 100.0

    rows = [
        ("CV $\\rightarrow$ XGBoost+nonlinear", "ADE", f"{100 * (cv_ade - xgb_nl) / cv_ade:.1f}\\%", "运动学先验之外，树模型残差显著降低 5 s 轨迹平均误差"),
        ("Ridge $\\rightarrow$ Ridge+nonlinear", "ADE", f"{100 * (ridge - ridge_nl) / ridge:.2f}\\%", "在线性残差模型中，非线性代理特征提供稳定补偿"),
        ("XGBoost $\\rightarrow$ XGBoost+nonlinear", "FDE", f"{100 * (fde_xgb - fde_xgb_nl) / fde_xgb:.2f}\\%", "强基学习器下增益较小，说明结论应保持克制"),
        ("10\\% 车道宽度噪声", "ADE", f"{robust:.1f} cm", "中高噪声下非线性融合略缓解观测扰动"),
        ("验证集融合权重", "$\\alpha_{XGB}$", f"{float(meta['xgboost_nonlinear_validation_blend_alpha']):.2f}", "非线性分支以校准权重进入最终预测，避免简单堆叠"),
    ]
    body = [
        "\\begin{tabularx}{\\textwidth}{llcX}",
        "\\toprule",
        "证据项 & 指标 & 数值 & 解释 \\\\",
        "\\midrule",
    ]
    body += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    body += ["\\bottomrule", "\\end{tabularx}"]
    (table_dir / "table_effect_summary.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def copy_to_report(run_fig_dir: Path, report_fig_dir: Path, table_dir: Path, ctx: dict[str, object]) -> None:
    ensure(report_fig_dir)
    for png in sorted(run_fig_dir.glob("fig*.png")):
        (report_fig_dir / png.name).write_bytes(png.read_bytes())
    write_evidence_table(table_dir, ctx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="code/runs/ngsim_8000_tracks")
    parser.add_argument("--report-fig-dir", default="figures")
    parser.add_argument("--table-dir", default="tables")
    args = parser.parse_args()

    configure_matplotlib()
    run_dir = Path(args.run_dir)
    fig_dir = ensure(run_dir / "figures_pub")
    ctx = load_context(run_dir)

    fig00_framework(fig_dir)
    fig01_tracks(fig_dir, ctx["tracks"])
    fig02_phase(fig_dir, ctx["tracks"])
    fig03_features(fig_dir, ctx["tracks"])
    fig04_metrics(fig_dir, ctx["metrics"])
    fig05_horizon(fig_dir, ctx["horizon"])
    fig06_ablation(fig_dir, ctx["ablation"])
    fig07_embedding(fig_dir, ctx["embed"])
    fig08_mse(fig_dir, ctx["mse"])
    fig09_noise(fig_dir, ctx["noise"])
    fig10_lane(fig_dir, ctx["cls"], run_dir)
    fig11_importance(fig_dir, ctx["imp"])
    fig12_cases(fig_dir, ctx["pred"])
    copy_to_report(fig_dir, Path(args.report_fig_dir), Path(args.table_dir), ctx)
    print(f"wrote publication figures to {fig_dir} and copied PNG files to {args.report_fig_dir}")


if __name__ == "__main__":
    main()
