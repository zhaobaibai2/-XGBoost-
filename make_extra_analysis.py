# -*- coding: utf-8 -*-
"""Create extra stratified-analysis tables and figures from saved predictions.

This script intentionally does not retrain models. It derives evaluation-only
diagnostics from the saved test predictions in a completed run directory:
scenario-style subsets, paired bootstrap confidence intervals, and grouped
XGBoost gain importance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def ade_fde(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y3 = y.reshape(-1, 50, 2)
    p3 = p.reshape(-1, 50, 2)
    d = np.linalg.norm(y3 - p3, axis=2)
    return d.mean(axis=1), d[:, -1]


def aggregate(y: np.ndarray, preds: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, pred in preds.items():
        a, f = ade_fde(y[mask], pred[mask])
        out[f"{name}_ADE"] = float(a.mean())
        out[f"{name}_FDE"] = float(f.mean())
    return out


def make_stratified(pred_path: Path, fig_dir: Path, table_dir: Path) -> None:
    data = np.load(pred_path)
    y = data["Yte"]
    y3 = y.reshape(-1, 50, 2)
    preds = {
        "XGBoost": data["XGBoost"],
        "XGBoost+NL": data["XGBoost_nonlinear"],
    }
    cv_ade, cv_fde = ade_fde(y, data["CV"])
    xgb_ade, xgb_fde = ade_fde(y, data["XGBoost"])

    lateral_range = y3[:, :, 0].max(axis=1) - y3[:, :, 0].min(axis=1)
    longitudinal = y3[:, :, 1]
    vel = np.gradient(longitudinal, 0.1, axis=1)
    acc = np.gradient(vel, 0.1, axis=1)

    masks = [
        ("All test", "全测试集", np.ones(len(y), dtype=bool)),
        ("Lane/high lateral", "换道/高横向扰动", lateral_range >= max(0.8 * 3.7, np.quantile(lateral_range, 0.75))),
        ("Hard braking", "急减速片段", acc.min(axis=1) <= np.quantile(acc.min(axis=1), 0.10)),
        ("High lon. variance", "高纵向波动", vel.std(axis=1) >= np.quantile(vel.std(axis=1), 0.75)),
        ("CV top20 FDE", "CV 高误差 top20\\%", cv_fde >= np.quantile(cv_fde, 0.80)),
        ("XGB top20 FDE", "XGBoost 高误差 top20\\%", xgb_fde >= np.quantile(xgb_fde, 0.80)),
    ]

    rows = []
    for scenario_en, scenario_cn, mask in masks:
        vals = aggregate(y, preds, mask)
        rows.append(
            {
                "场景": scenario_cn,
                "Scenario": scenario_en,
                "样本数": int(mask.sum()),
                "XGB ADE": vals["XGBoost_ADE"],
                "XGB+NL ADE": vals["XGBoost+NL_ADE"],
                "XGB FDE": vals["XGBoost_FDE"],
                "XGB+NL FDE": vals["XGBoost+NL_FDE"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(table_dir / "stratified_scenarios.csv", index=False)

    body = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "场景 & 样本数 & XGB ADE & XGB+NL ADE & XGB FDE & XGB+NL FDE \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        body.append(
            f"{r['场景']} & {int(r['样本数'])} & {r['XGB ADE']:.3f} & {r['XGB+NL ADE']:.3f} & {r['XGB FDE']:.3f} & {r['XGB+NL FDE']:.3f} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}"]
    (table_dir / "table_stratified_scenarios.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(3.94, 2.35))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["XGB FDE"], width=width, label="XGBoost", color="#B07AA1")
    ax.bar(x + width / 2, df["XGB+NL FDE"], width=width, label="XGBoost+NL", color="#D95F02")
    ax.set_ylabel("FDE (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Scenario"], rotation=28, ha="right")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="#E5E7EB", lw=0.4)
    savefig(fig, fig_dir, "fig13_stratified_scenarios")


def make_baseline_matrix(run_dir: Path, table_dir: Path) -> None:
    metrics = pd.read_csv(run_dir / "results" / "prediction_metrics.csv")
    keep = [
        "CV", "CA", "Ridge", "Ridge-nonlinear-only", "Ridge+nonlinear",
        "XGBoost", "XGBoost-nonlinear-only", "XGBoost+nonlinear",
    ]
    metrics = metrics[metrics["Model"].isin(keep)].copy()
    metrics["Model"] = pd.Categorical(metrics["Model"], keep, ordered=True)
    metrics = metrics.sort_values("Model")
    body = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "方法 & ADE/m & FDE/m & RMSE/m & Lane F1 \\\\",
        "\\midrule",
    ]
    for _, r in metrics.iterrows():
        body.append(f"{r['Model']} & {r['ADE']:.3f} & {r['FDE']:.3f} & {r['RMSE']:.3f} & {r['LaneChange_F1']:.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (table_dir / "table_baseline_matrix_results.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    cand_path = run_dir / "results" / "xgboost_nonlinear_candidates.csv"
    if cand_path.exists():
        cand = pd.read_csv(cand_path)
        body = ["\\begin{tabular}{lccc}", "\\toprule", "候选非线性分支 & 融合权重 $\\alpha$ & Val ADE/m & Val FDE/m \\\\", "\\midrule"]
        for _, r in cand.iterrows():
            body.append(f"{r['Candidate']} & {r['Alpha']:.2f} & {r['Val_ADE']:.3f} & {r['Val_FDE']:.3f} \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        (table_dir / "table_xgb_candidates.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def make_bootstrap(pred_path: Path, fig_dir: Path, table_dir: Path, n_boot: int = 1000) -> None:
    data = np.load(pred_path)
    y = data["Yte"]
    a_x, f_x = ade_fde(y, data["XGBoost"])
    a_n, f_n = ade_fde(y, data["XGBoost_nonlinear"])
    diff_ade = a_x - a_n
    diff_fde = f_x - f_n

    rng = np.random.default_rng(42)
    rows = []
    plot_rows = []
    for metric, diff in [("ADE", diff_ade), ("FDE", diff_fde)]:
        samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(diff), size=len(diff))
            samples.append(float(diff[idx].mean()))
        lo, hi = np.percentile(samples, [2.5, 97.5])
        mean = float(diff.mean())
        rows.append((metric, mean, float(lo), float(hi), "是" if lo > 0 or hi < 0 else "否"))
        plot_rows.append((metric, mean, float(lo), float(hi)))

    body = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "指标 & 平均差值(XGB-XGB+NL) & 95\\% CI & 显著 \\\\",
        "\\midrule",
    ]
    for metric, mean, lo, hi, sig in rows:
        body.append(f"{metric} & {mean:.4f} & [{lo:.4f}, {hi:.4f}] & {sig} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (table_dir / "table_bootstrap.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(2.7, 1.8))
    yloc = np.arange(len(plot_rows))
    means = np.array([r[1] for r in plot_rows])
    los = np.array([r[2] for r in plot_rows])
    his = np.array([r[3] for r in plot_rows])
    ax.errorbar(means, yloc, xerr=[means - los, his - means], fmt="o", color="#D95F02", capsize=3)
    ax.axvline(0, color="#6B7280", lw=0.8, ls="--")
    ax.set_yticks(yloc)
    ax.set_yticklabels([r[0] for r in plot_rows])
    ax.set_xlabel("Error reduction (m)")
    ax.grid(axis="x", color="#E5E7EB", lw=0.4)
    savefig(fig, fig_dir, "fig14_bootstrap_ci")


def group_feature(name: str) -> str:
    if name.startswith("sq_"):
        return group_feature(name[3:])
    if name.startswith("hist_") or name in {"vx_last", "vy_last", "vx_mean", "vy_mean", "vx_std", "vy_std", "ax_mean", "ay_mean", "ay_last"}:
        return "Kinematic history"
    if name.startswith("delay_"):
        return "Delay embedding"
    if "lyapunov" in name or "jerk" in name:
        return "Local divergence"
    if "rqa" in name or name in {"rr", "det"}:
        return "RQA"
    if "sampen" in name or "mse" in name or "dfa" in name or "hurst" in name:
        return "Entropy/DFA"
    if "graph" in name or "density" in name or "headway" in name or "neighbor" in name:
        return "Neighbor graph"
    return "Other"


def make_group_importance(imp_path: Path, fig_dir: Path, table_dir: Path) -> None:
    imp = pd.read_csv(imp_path)
    imp["Group"] = imp["Feature"].map(group_feature)
    g = imp.groupby("Group", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)
    g.to_csv(table_dir / "feature_group_importance.csv", index=False)
    body = ["\\begin{tabular}{lr}", "\\toprule", "特征组 & XGBoost gain 占比 \\\\", "\\midrule"]
    total = float(g["Importance"].sum())
    for _, r in g.iterrows():
        body.append(f"{r['Group']} & {100.0 * r['Importance'] / total:.1f}\\% \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (table_dir / "table_feature_groups.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.barh(g["Group"], 100.0 * g["Importance"] / total, color="#4C9A8A")
    ax.invert_yaxis()
    ax.set_xlabel("Gain share (%)")
    ax.grid(axis="x", color="#E5E7EB", lw=0.4)
    savefig(fig, fig_dir, "fig15_feature_group_importance")


def make_permutation_importance(run_dir: Path, fig_dir: Path, table_dir: Path) -> None:
    path = run_dir / "results" / "permutation_importance.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("Importance", ascending=False).head(12)
    body = ["\\begin{tabular}{lr}", "\\toprule", "特征 & ADE 增量/m \\\\", "\\midrule"]
    for _, r in df.iterrows():
        feat = str(r["Feature"]).replace("_", "\\_")
        body.append(f"{feat} & {r['Importance']:.5f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (table_dir / "table_permutation_importance.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    plot_df = df.iloc[::-1]
    ax.barh(plot_df["Feature"], plot_df["Importance"], color="#D95F02")
    ax.set_xlabel("ADE increase (m)")
    ax.grid(axis="x", color="#E5E7EB", lw=0.4)
    savefig(fig, fig_dir, "fig16_permutation_importance")


def make_sensitivity_figures(run_dir: Path, fig_dir: Path, table_dir: Path) -> None:
    res = run_dir / "results"
    mtau_path = res / "sensitivity_m_tau.csv"
    if mtau_path.exists():
        mtau = pd.read_csv(mtau_path)
        pivot = mtau.pivot(index="Embedding_dim", columns="Tau", values="ADE")
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(r"$\tau$ (frames)")
        ax.set_ylabel(r"$m$")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("ADE (m)")
        savefig(fig, fig_dir, "fig17_m_tau_heatmap")

    samp_path = res / "sensitivity_sampen_r.csv"
    rqa_path = res / "sensitivity_rqa_eps.csv"
    if samp_path.exists() and rqa_path.exists():
        samp = pd.read_csv(samp_path)
        rqa = pd.read_csv(rqa_path)
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))
        axes[0].plot(samp["R_ratio"], samp["ADE"], marker="o", color="#3B6EA8")
        axes[0].set_xlabel(r"SampEn $r/\sigma$")
        axes[0].set_ylabel("ADE (m)")
        axes[0].grid(True, color="#E5E7EB", lw=0.4)
        axes[1].plot(rqa["RQA_eps_percentile"], rqa["ADE"], marker="o", color="#4C9A8A")
        axes[1].set_xlabel("RQA eps percentile")
        axes[1].set_ylabel("ADE (m)")
        axes[1].grid(True, color="#E5E7EB", lw=0.4)
        savefig(fig, fig_dir, "fig18_entropy_rqa_sensitivity")

        body = ["\\begin{tabular}{lccc}", "\\toprule", "参数 & 取值范围 & 最低 ADE/m & 波动范围/m \\\\", "\\midrule"]
        body.append(f"SampEn $r$ & 0.10--0.30$\\sigma$ & {samp['ADE'].min():.3f} & {samp['ADE'].max() - samp['ADE'].min():.3f} \\\\")
        body.append(f"RQA $\\varepsilon$ & 5--20\\% 分位数 & {rqa['ADE'].min():.3f} & {rqa['ADE'].max() - rqa['ADE'].min():.3f} \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        (table_dir / "table_parameter_sensitivity.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="code/runs/ngsim_8000_tracks")
    ap.add_argument("--report-fig-dir", default="figures")
    ap.add_argument("--table-dir", default="tables")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = ensure(Path(args.report_fig_dir))
    table_dir = ensure(Path(args.table_dir))
    make_baseline_matrix(run_dir, table_dir)
    make_stratified(run_dir / "results" / "predictions_test.npz", fig_dir, table_dir)
    make_bootstrap(run_dir / "results" / "predictions_test.npz", fig_dir, table_dir)
    make_group_importance(run_dir / "results" / "feature_importance.csv", fig_dir, table_dir)
    make_permutation_importance(run_dir, fig_dir, table_dir)
    make_sensitivity_figures(run_dir, fig_dir, table_dir)
    print("wrote extra analysis figures and tables")


if __name__ == "__main__":
    main()
