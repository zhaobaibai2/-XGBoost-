# -*- coding: utf-8 -*-
"""
Course-report reproducible experiment:
Nonlinear feature + intelligent model fusion for autonomous-driving trajectory prediction.

Main outputs:
  data/generated/ngsim_compatible_benchmark.csv
  results/*.csv, results/*.json
  tables/*.tex
  figures/*.png

The script is GPU-first. When CUDA is visible, XGBoost is run with device='cuda'.
In this sandbox CUDA is not exposed; the script falls back to CPU and records the
reason in results/gpu_status.json.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

DT = 0.1
HIST = 30
PRED = 50
LANE_W = 3.7
FT_TO_M = 0.3048
SAMPEN_R_RATIO = 0.2
RQA_EPS_PERCENTILE = 15.0


def ensure(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_acceleration(requested: str = "auto") -> dict:
    status = {
        "requested_device": requested,
        "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
        "nvidia_smi_output": None,
        "torch_cuda_available": False,
        "torch_version": None,
        "xgboost_version": getattr(xgb, "__version__", None) if xgb is not None else None,
        "device_selected": "cpu",
        "fallback_reason": "CUDA was not requested or not detected."
    }
    if status["nvidia_smi_found"]:
        try:
            r = subprocess.run(["nvidia-smi"], text=True, capture_output=True, timeout=6)
            status["nvidia_smi_output"] = (r.stdout + r.stderr).strip()[:2000]
        except Exception as e:
            status["nvidia_smi_output"] = repr(e)
    try:
        import torch
        status["torch_version"] = torch.__version__
        status["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        status["torch_version"] = f"not importable: {e!r}"

    if requested == "cpu":
        status["device_selected"] = "cpu"
        status["fallback_reason"] = "User selected CPU."
    elif requested == "cuda":
        status["device_selected"] = "cuda"
        status["fallback_reason"] = "User forced CUDA; fit will fall back to CPU only if CUDA fails."
    elif status["nvidia_smi_found"] or status["torch_cuda_available"]:
        status["device_selected"] = "cuda"
        status["fallback_reason"] = "CUDA detected."
    else:
        status["device_selected"] = "cpu"
        status["fallback_reason"] = "No nvidia-smi command and torch.cuda.is_available() is False."
    return status


# -----------------------------------------------------------------------------
# Data generation and loading
# -----------------------------------------------------------------------------
def simulate_ngsim_compatible(seed: int = 42, n: int = 56, T: int = 620, lanes: int = 4):
    """Generate a reproducible NGSIM-field-compatible nonlinear benchmark in metres."""
    rng = np.random.default_rng(seed)
    centers = np.arange(lanes) * LANE_W + LANE_W / 2
    x = np.zeros((n, T))
    y = np.zeros((n, T))
    v = np.zeros((n, T))
    a = np.zeros((n, T))
    lane = np.zeros((n, T), dtype=int)

    desired = rng.uniform(20.0, 31.0, size=n)
    phase = rng.uniform(0, 2*np.pi, size=n)
    for i in range(n):
        lane[i, 0] = rng.integers(0, lanes)
        x[i, 0] = centers[lane[i, 0]] + rng.normal(0, 0.08)
        y[i, 0] = -i * 10.5 + rng.normal(0, 2.5)
        v[i, 0] = rng.uniform(14.0, 25.5)

    change_plan = {}
    first_low, first_high = 100, max(101, T-165)
    second_low, second_high = 180, max(181, T-90)
    for i in range(n):
        times = []
        if first_high > first_low and rng.random() < 0.65:
            times.append(int(rng.integers(first_low, first_high)))
        if second_high > second_low and rng.random() < 0.24:
            times.append(int(rng.integers(second_low, second_high)))
        change_plan[i] = sorted(times)

    target_lane = lane[:, 0].copy()
    start_x = x[:, 0].copy()
    target_x = x[:, 0].copy()
    start_t = np.full(n, -999)
    duration = 46

    for t in range(1, T):
        prev_lane = np.clip(np.round((x[:, t-1] - LANE_W/2) / LANE_W), 0, lanes-1).astype(int)

        for i in range(n):
            if change_plan[i] and t == change_plan[i][0]:
                change_plan[i].pop(0)
                candidates = []
                if prev_lane[i] > 0:
                    candidates.append(prev_lane[i]-1)
                if prev_lane[i] < lanes-1:
                    candidates.append(prev_lane[i]+1)
                if candidates:
                    target_lane[i] = int(rng.choice(candidates))
                    start_x[i] = x[i, t-1]
                    target_x[i] = centers[target_lane[i]]
                    start_t[i] = t

        for i in range(n):
            if 0 <= t - start_t[i] <= duration:
                u = (t - start_t[i]) / duration
                s = 1.0 / (1.0 + np.exp(-10*(u-0.5)))
                # Small oscillatory term makes a lane change not perfectly deterministic.
                x[i, t] = start_x[i] + (target_x[i] - start_x[i]) * s + 0.035*np.sin(0.13*t + phase[i]) + rng.normal(0, 0.025)
            else:
                x[i, t] = x[i, t-1] + 0.025*np.sin(0.035*t + phase[i]) + rng.normal(0, 0.025)

        curr_lane = np.clip(np.round((x[:, t-1] - LANE_W/2) / LANE_W), 0, lanes-1).astype(int)

        for i in range(n):
            same_lane = np.where(curr_lane == curr_lane[i])[0]
            ahead = same_lane[y[same_lane, t-1] > y[i, t-1] + 1.0]
            if len(ahead):
                j = ahead[np.argmin(y[ahead, t-1] - y[i, t-1])]
                gap = max(y[j, t-1] - y[i, t-1] - 4.8, 0.4)
                dv = v[i, t-1] - v[j, t-1]
            else:
                gap, dv = 90.0, 0.0

            amax, b, th, s0, delta = 1.65, 2.20, 1.10, 3.0, 4.0
            s_star = s0 + max(0.0, v[i, t-1]*th + v[i, t-1]*dv/(2*np.sqrt(amax*b)))
            idm = amax * (1 - (v[i, t-1]/desired[i])**delta - (s_star/gap)**2)
            wave = 0.44*np.sin(0.055*t + phase[i]) + 0.24*np.sin(0.011*y[i, t-1] + 0.35*curr_lane[i])
            lateral_coupling = -0.12 * abs(x[i, t-1] - centers[curr_lane[i]])
            random_brake = -0.9 if (rng.random() < 0.004 and len(ahead)) else 0.0
            a[i, t] = np.clip(idm + wave + lateral_coupling + random_brake, -4.5, 2.4)
            v[i, t] = max(0.2, v[i, t-1] + a[i, t]*DT)
            y[i, t] = y[i, t-1] + v[i, t]*DT + 0.5*a[i, t]*DT*DT + rng.normal(0, 0.045)
            lane[i, t] = np.clip(np.round((x[i, t] - LANE_W/2) / LANE_W), 0, lanes-1).astype(int)

    return x, y, v, a, lane


def dataframe_from_matrices(x, y, v, a, lane):
    n, T = x.shape
    frames = []
    for i in range(n):
        frames.append(pd.DataFrame({
            "Vehicle_ID": i + 1,
            "Frame_ID": np.arange(1, T + 1),
            "Total_Frames": T,
            "Global_Time": (np.arange(T) * 100).astype(int),
            "Local_X": x[i],
            "Local_Y": y[i],
            "v_Vel": v[i],
            "v_Acc": a[i],
            "Lane_ID": lane[i] + 1,
            "Preceding": 0,
            "Following": 0,
            "Space_Headway": np.nan,
            "Location": "synthetic-ngsim-compatible"
        }))
    return pd.concat(frames, ignore_index=True)


def load_real_ngsim_csv(path: str | Path, input_units: str = "feet"):
    """Load USDOT NGSIM-like CSV. Official NGSIM coordinates are in feet; default converts to metres."""
    df = pd.read_csv(path)
    lower = {str(c).lower(): c for c in df.columns}
    mapping = {
        "vehicle_id": "Vehicle_ID", "frame_id": "Frame_ID", "total_frames": "Total_Frames", "global_time": "Global_Time",
        "local_x": "Local_X", "local_y": "Local_Y", "global_x": "Global_X", "global_y": "Global_Y",
        "v_vel": "v_Vel", "v_acc": "v_Acc", "lane_id": "Lane_ID", "preceding": "Preceding",
        "following": "Following", "space_headway": "Space_Headway", "location": "Location"
    }
    rename = {}
    for low, canon in mapping.items():
        if canon not in df.columns and low in lower:
            rename[lower[low]] = canon
    df = df.rename(columns=rename)
    required = ["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from CSV: {missing}")

    for c in ["Vehicle_ID", "Frame_ID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y"]).copy()
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Frame_ID"] = df["Frame_ID"].astype(int)

    if "Global_Time" not in df.columns:
        df["Global_Time"] = df["Frame_ID"] * 100
    if "Lane_ID" not in df.columns:
        df["Lane_ID"] = np.clip(np.round(df["Local_X"] / LANE_W) + 1, 1, 8).astype(int)
    if "v_Vel" not in df.columns:
        df["v_Vel"] = df.groupby("Vehicle_ID")["Local_Y"].diff().fillna(0) / DT
    if "v_Acc" not in df.columns:
        df["v_Acc"] = df.groupby("Vehicle_ID")["v_Vel"].diff().fillna(0) / DT
    for c in ["Preceding", "Following"]:
        if c not in df.columns:
            df[c] = 0
    if "Space_Headway" not in df.columns:
        df["Space_Headway"] = np.nan

    if input_units == "feet":
        for c in ["Local_X", "Local_Y", "v_Vel"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * FT_TO_M
        if "v_Acc" in df.columns:
            df["v_Acc"] = pd.to_numeric(df["v_Acc"], errors="coerce") * FT_TO_M
        if "Space_Headway" in df.columns:
            df["Space_Headway"] = pd.to_numeric(df["Space_Headway"], errors="coerce") * FT_TO_M
    elif input_units != "meters":
        raise ValueError("input_units must be 'feet' or 'meters'.")

    return df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)


def matrix_from_df(df: pd.DataFrame, max_frames: int = 620):
    groups = []
    min_len = None
    for vid, g in df.groupby("Vehicle_ID"):
        g = g.sort_values("Frame_ID")
        if len(g) >= HIST + PRED + 5:
            groups.append((vid, g))
            min_len = len(g) if min_len is None else min(min_len, len(g))
    if not groups:
        raise ValueError("No vehicle has enough continuous frames for the selected windows.")
    n = len(groups)
    T = min(int(min_len), max_frames)
    x = np.zeros((n, T)); y = np.zeros((n, T)); v = np.zeros((n, T)); a = np.zeros((n, T)); lane = np.zeros((n, T), dtype=int)
    vehicle_ids = []
    for i, (vid, g) in enumerate(groups):
        gg = g.iloc[:T]
        vehicle_ids.append(int(vid))
        x[i] = pd.to_numeric(gg["Local_X"], errors="coerce").interpolate().bfill().ffill().to_numpy(float)
        y[i] = pd.to_numeric(gg["Local_Y"], errors="coerce").interpolate().bfill().ffill().to_numpy(float)
        v[i] = pd.to_numeric(gg["v_Vel"], errors="coerce").interpolate().bfill().ffill().to_numpy(float)
        a[i] = pd.to_numeric(gg["v_Acc"], errors="coerce").interpolate().bfill().ffill().to_numpy(float)
        lane[i] = pd.to_numeric(gg["Lane_ID"], errors="coerce").fillna(method="ffill").fillna(1).to_numpy(int) - 1
    return np.array(vehicle_ids), x, y, v, a, lane


# -----------------------------------------------------------------------------
# Nonlinear features
# -----------------------------------------------------------------------------
def _embed(series: np.ndarray, m: int = 3, tau: int = 4):
    y = np.asarray(series, dtype=float)
    rows = []
    for k in range((m-1)*tau, len(y)):
        rows.append([y[k - j*tau] for j in range(m)])
    return np.asarray(rows, dtype=float)


def recurrence_matrix(series: np.ndarray, m: int = 3, tau: int = 4):
    E = _embed(series, m=m, tau=tau)
    if len(E) < 3:
        return np.zeros((0, 0), dtype=bool)
    D = np.sqrt(((E[:, None, :] - E[None, :, :])**2).sum(axis=2))
    nz = D[D > 0]
    eps = np.percentile(nz, RQA_EPS_PERCENTILE) if len(nz) else 1.0
    R = (D < eps)
    np.fill_diagonal(R, False)
    return R


def rqa_metrics(series: np.ndarray, m: int = 3, tau: int = 4):
    R = recurrence_matrix(series, m=m, tau=tau)
    if R.size == 0:
        return 0.0, 0.0, 0.0
    rec_points = int(R.sum())
    rr = rec_points / max(1, R.size - R.shape[0])
    diag_points = 0
    diag_lengths = []
    n = R.shape[0]
    for offset in range(-(n-1), n):
        diag = np.diag(R, k=offset)
        run = 0
        for val in diag:
            if val:
                run += 1
            else:
                if run >= 2:
                    diag_points += run
                    diag_lengths.append(run)
                run = 0
        if run >= 2:
            diag_points += run
            diag_lengths.append(run)
    det = diag_points / max(1, rec_points)
    mean_diag = float(np.mean(diag_lengths)) if diag_lengths else 0.0
    return float(rr), float(det), mean_diag


def lyapunov_proxy(series: np.ndarray, m: int = 3, tau: int = 4):
    E = _embed(series, m=m, tau=tau)
    if len(E) < 5:
        return 0.0
    vals = []
    for i in range(len(E)-1):
        d = np.sqrt(((E - E[i])**2).sum(axis=1))
        d[max(0, i-2):min(len(E), i+3)] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or j >= len(E)-1:
            continue
        d0 = np.linalg.norm(E[i] - E[j])
        d1 = np.linalg.norm(E[i+1] - E[j+1])
        if d0 > 1e-9 and d1 > 1e-9:
            vals.append(np.log(d1/d0))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), -4.0, 4.0))


def sample_entropy(series: np.ndarray, m: int = 2, r_ratio: float | None = None):
    """Vectorized sample entropy for short trajectory windows."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    sd = float(np.std(x))
    if n <= m + 2 or sd < 1e-12:
        return 0.0
    if r_ratio is None:
        r_ratio = SAMPEN_R_RATIO
    r = r_ratio * sd

    def count(mm: int):
        T = sliding_window_view(x, mm)
        D = np.max(np.abs(T[:, None, :] - T[None, :, :]), axis=2)
        # remove self-matches and count unordered pairs
        return int((np.sum(D <= r) - len(T)) // 2)

    B = count(m)
    A = count(m+1)
    return float(np.log((B + 1.0) / (A + 1.0)))


def multiscale_sampen(series: np.ndarray, scales: Sequence[int] = (1, 2, 3, 4, 5)):
    x = np.asarray(series, dtype=float)
    vals = []
    for s in scales:
        usable = len(x) // s * s
        if usable < 8:
            vals.append(np.nan)
            continue
        cg = x[:usable].reshape(-1, s).mean(axis=1)
        vals.append(sample_entropy(cg))
    return np.asarray(vals, dtype=float)


def dfa_alpha(series: np.ndarray):
    x = np.asarray(series, dtype=float)
    if len(x) < 12 or np.std(x) < 1e-12:
        return 0.5
    y = np.cumsum(x - np.mean(x))
    sizes = np.array([4, 5, 6, 8, 10, 12, 15])
    Fs = []
    used = []
    for s in sizes:
        nseg = len(y) // s
        if nseg < 2:
            continue
        rms = []
        for k in range(nseg):
            seg = y[k*s:(k+1)*s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, 1)
            trend = coef[0]*t + coef[1]
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F = float(np.sqrt(np.mean(np.square(rms))))
        if F > 1e-12:
            used.append(s)
            Fs.append(F)
    if len(Fs) < 2:
        return 0.5
    alpha = np.polyfit(np.log(used), np.log(Fs), 1)[0]
    return float(np.clip(alpha, 0.0, 2.0))


def finite_complexity_proxy(speed: np.ndarray):
    ds = np.diff(speed)
    return float(np.log1p(np.std(ds)) / (np.mean(np.abs(ds)) + 1e-4))


# -----------------------------------------------------------------------------
# Samples and features
# -----------------------------------------------------------------------------
def make_samples(vehicle_ids, x, y, v, a, lane, stride: int = 12):
    n, T = x.shape
    samples = []
    for i in range(n):
        for s in range(0, T-HIST-PRED, stride):
            hist_xy = np.c_[x[i, s:s+HIST], y[i, s:s+HIST]]
            fut_xy = np.c_[x[i, s+HIST:s+HIST+PRED], y[i, s+HIST:s+HIST+PRED]]
            t = s + HIST - 1
            dx = x[:, t] - x[i, t]
            dy = y[:, t] - y[i, t]
            dist = np.sqrt(dx*dx + dy*dy)
            mask = (np.arange(n) != i) & (dist < 60.0)
            ahead = mask & (y[:, t] > y[i, t])
            min_head = float(np.min(y[ahead, t] - y[i, t])) if ahead.any() else 85.0
            graph = np.array([
                mask.sum()/60.0,
                min(min_head, 85.0),
                float(np.mean(v[mask, t] - v[i, t]) if mask.any() else 0.0),
                float(((lane[mask, t] == lane[i, t]).sum()/60.0) if mask.any() else 0.0),
                float(mask.sum())
            ], dtype=float)
            samples.append({
                "vehicle_id": int(vehicle_ids[i]),
                "start_frame": int(s+1),
                "hist_xy": hist_xy,
                "future_xy": fut_xy,
                "hist_lane": lane[i, s:s+HIST],
                "future_lane": lane[i, s+HIST:s+HIST+PRED],
                "hist_acc": a[i, s:s+HIST],
                "graph": graph
            })
    return samples


def add_poly_features(names: list[str], vals: list[float], max_tail: int = 20):
    tail_names = names[-min(max_tail, len(names)):]
    tail_vals = vals[-min(max_tail, len(vals)):]
    for name, val in zip(tail_names, tail_vals):
        names.append(f"sq_{name}")
        vals.append(float(val*val))


def feature_one(sample: dict, groups: Iterable[str], m: int = 3, tau: int = 4, noise: float = 0.0, rng=None):
    groups = set(groups)
    h = sample["hist_xy"].copy().astype(float)
    if noise > 0:
        if rng is None:
            rng = np.random.default_rng(1)
        h = h + rng.normal(0.0, noise, h.shape)
    xh, yh = h[:, 0], h[:, 1]
    vx = np.gradient(xh, DT)
    vy = np.gradient(yh, DT)
    ax = np.gradient(vx, DT)
    ay = np.gradient(vy, DT)
    speed = np.sqrt(vx*vx + vy*vy)
    names, vals = [], []

    def add(name, val):
        names.append(name)
        vals.append(float(0.0 if not np.isfinite(val) else val))

    if "base" in groups:
        last = h[-1]
        add("last_x", last[0]); add("last_y", last[1])
        for j in range(12, 0, -1):
            add(f"hist_dx_t-{j}", xh[-j] - last[0])
            add(f"hist_dy_t-{j}", yh[-j] - last[1])
        for name, arr in [("vx", vx), ("vy", vy), ("ax", ax), ("ay", ay)]:
            add(f"{name}_last", arr[-1])
            add(f"{name}_mean", np.mean(arr))
            add(f"{name}_std", np.std(arr))
        add("lat_range", np.max(xh) - np.min(xh))
        add("lon_displacement", yh[-1] - yh[0])
        add("hist_lane_last", sample["hist_lane"][-1])
        add("hist_lane_change_count", np.sum(np.diff(sample["hist_lane"]) != 0))

    if "delay" in groups:
        for k in range(m):
            idx = max(0, len(yh)-1-k*tau)
            add(f"delay_y_{k}", yh[idx] - yh[-1])
            add(f"delay_x_{k}", xh[idx] - xh[-1])

    if "lyap" in groups:
        add("lyapunov_proxy_y", lyapunov_proxy(yh, m=max(2, min(m, 6)), tau=tau))
        add("complexity_speed", finite_complexity_proxy(speed))
        jerk_y = np.gradient(ay, DT)
        add("jerk_y_std", np.std(jerk_y))
        add("acc_abs_p90", np.percentile(np.abs(ay), 90))

    if "rqa" in groups:
        rr, det, lmean = rqa_metrics(yh, m=max(2, min(m, 6)), tau=tau)
        add("rqa_recurrence_rate", rr)
        add("rqa_determinism", det)
        add("rqa_mean_diag_len", lmean)

    if "entropy" in groups:
        se_y = sample_entropy(yh)
        se_v = sample_entropy(speed)
        mse = multiscale_sampen(speed, scales=(1, 2, 3, 4, 5))
        mse_filled = np.nan_to_num(mse, nan=np.nanmean(mse[np.isfinite(mse)]) if np.any(np.isfinite(mse)) else 0.0)
        add("sampen_y", se_y)
        add("sampen_speed", se_v)
        for idx, val in enumerate(mse_filled, start=1):
            add(f"mse_speed_s{idx}", val)
        add("mse_speed_mean", np.mean(mse_filled))
        valid = np.where(np.isfinite(mse))[0]
        if len(valid) >= 2:
            slope = np.polyfit(valid + 1, mse[valid], 1)[0]
        else:
            slope = 0.0
        add("mse_speed_slope", slope)
        add("dfa_y_alpha", dfa_alpha(yh))
        add("dfa_speed_alpha", dfa_alpha(speed))

    if "graph" in groups:
        labels = ["density", "min_headway", "mean_rel_speed", "same_lane_density", "neighbor_count"]
        for lab, val in zip(labels, sample["graph"]):
            add(f"graph_{lab}", val)

    if "poly" in groups:
        add_poly_features(names, vals)

    return names, vals


def feature_matrix(samples: Sequence[dict], groups: Iterable[str], m: int = 3, tau: int = 4, noise: float = 0.0, seed: int = 1, return_names: bool = False):
    rng = np.random.default_rng(seed)
    rows = []
    names_ref = None
    for s in samples:
        names, vals = feature_one(s, groups=groups, m=m, tau=tau, noise=noise, rng=rng)
        if names_ref is None:
            names_ref = names
        rows.append(vals)
    maxlen = max(len(r) for r in rows)
    X = np.array([r + [0.0]*(maxlen-len(r)) for r in rows], dtype=float)
    if return_names:
        if len(names_ref) < maxlen:
            names_ref = names_ref + [f"pad_{i}" for i in range(maxlen-len(names_ref))]
        return X, names_ref
    return X


def targets(samples: Sequence[dict]):
    Y = np.array([s["future_xy"].ravel() for s in samples], dtype=float)
    C = np.array([int(np.any(s["future_lane"] != s["hist_lane"][-1])) for s in samples], dtype=int)
    return Y, C


def split_samples(samples: Sequence[dict], seed: int = 42):
    rng = np.random.default_rng(seed)
    vids = np.array(sorted({s["vehicle_id"] for s in samples}))
    if len(vids) >= 10:
        rng.shuffle(vids)
        n_train = max(1, int(0.70*len(vids)))
        n_val = max(1, int(0.15*len(vids)))
        train_vids = set(vids[:n_train])
        val_vids = set(vids[n_train:n_train+n_val])
        test_vids = set(vids[n_train+n_val:])
        train = [s for s in samples if s["vehicle_id"] in train_vids]
        val = [s for s in samples if s["vehicle_id"] in val_vids]
        test = [s for s in samples if s["vehicle_id"] in test_vids]
    else:
        order = rng.permutation(len(samples))
        a = int(0.70*len(order)); b = int(0.85*len(order))
        train = [samples[i] for i in order[:a]]
        val = [samples[i] for i in order[a:b]]
        test = [samples[i] for i in order[b:]]
    return train, val, test



# -----------------------------------------------------------------------------
# Smooth residual basis for fast XGBoost training
# -----------------------------------------------------------------------------
def residual_basis_matrix():
    u = np.linspace(1.0/PRED, 1.0, PRED)
    B = np.vstack([u, u*u, u*u*u]).T
    return B


def residual_to_basis(R):
    """Project a 2*PRED residual trajectory to six smooth basis coefficients."""
    B = residual_basis_matrix()
    pinv = np.linalg.pinv(B)
    rr = R.reshape(-1, PRED, 2)
    coefs = []
    for d in range(2):
        coefs.append(rr[:, :, d] @ pinv.T)
    return np.concatenate(coefs, axis=1)


def basis_to_residual(C):
    """Expand six basis coefficients back to a full 2*PRED residual trajectory."""
    B = residual_basis_matrix()
    C = np.asarray(C, dtype=float)
    k = B.shape[1]
    cx = C[:, :k]
    cy = C[:, k:2*k]
    rx = cx @ B.T
    ry = cy @ B.T
    R = np.stack([rx, ry], axis=2)
    return R.reshape(len(C), 2*PRED)

# -----------------------------------------------------------------------------
# Models and metrics
# -----------------------------------------------------------------------------
class StandardRidge:
    def __init__(self, lam: float = 10.0):
        self.lam = float(lam)

    def fit(self, X, Y):
        self.n_features_ = X.shape[1]
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        Z = np.c_[np.ones(len(Z)), Z]
        I = np.eye(Z.shape[1]); I[0, 0] = 0
        self.W = np.linalg.solve(Z.T @ Z + self.lam * I, Z.T @ Y)
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.c_[np.ones(len(Z)), Z]
        return Z @ self.W


class WeightedLogistic:
    def fit(self, X, y, lr: float = 0.06, epochs: int = 900, lam: float = 0.004):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        Z = np.c_[np.ones(len(Z)), Z]
        w = np.zeros(Z.shape[1])
        pos = max(1, int(y.sum()))
        neg = max(1, int(len(y) - y.sum()))
        weights = np.where(y == 1, neg/pos, 1.0)
        weights = weights / np.mean(weights)
        for _ in range(epochs):
            p = 1/(1+np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ ((p-y)*weights) / len(y) + lam*np.r_[0, w[1:]]
            w -= lr * grad
        self.w = w
        return self

    def proba(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.c_[np.ones(len(Z)), Z]
        return 1/(1+np.exp(-np.clip(Z @ self.w, -30, 30)))


class StandardXGBRegressor:
    def __init__(self, seed: int = 42, device: str = "cpu", n_estimators: int = 160, max_depth: int = 3, learning_rate: float = 0.055):
        if xgb is None:
            raise RuntimeError("xgboost is not installed.")
        self.seed = seed
        self.device = device
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.used_device = device
        self.fallback_note = ""

    def _new_model(self, device):
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.92,
            colsample_bytree=0.90,
            reg_lambda=3.0,
            min_child_weight=2.0,
            random_state=self.seed,
            tree_method="hist",
            device=device,
            n_jobs=max(1, min(8, os.cpu_count() or 1)),
            multi_strategy="multi_output_tree"
        )

    def fit(self, X, Y):
        self.n_features_ = X.shape[1]
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        try:
            self.model = self._new_model(self.device)
            self.model.fit(Z, Y, verbose=False)
            self.used_device = self.device
            return self
        except Exception as e:
            if self.device == "cuda":
                warnings.warn(f"XGBoost CUDA fit failed; falling back to CPU: {e}")
                self.model = self._new_model("cpu")
                self.model.fit(Z, Y, verbose=False)
                self.used_device = "cpu"
                self.fallback_note = repr(e)[:500]
                return self
            raise

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        return np.asarray(self.model.predict(Z), dtype=float)

    def importance(self):
        try:
            imp = getattr(self.model, "feature_importances_", None)
            if imp is not None:
                return np.asarray(imp, dtype=float)
        except Exception:
            pass
        try:
            score = self.model.get_booster().get_score(importance_type="weight")
            nfeat = getattr(self, "n_features_", 0)
            imp = np.zeros(nfeat, dtype=float)
            for key, val in score.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if idx < len(imp):
                        imp[idx] = float(val)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            return imp
        except Exception:
            return None


class StandardXGBClassifier:
    def __init__(self, seed: int = 42, device: str = "cpu", n_estimators: int = 180):
        if xgb is None:
            raise RuntimeError("xgboost is not installed.")
        self.seed = seed
        self.device = device
        self.n_estimators = n_estimators
        self.used_device = device
        self.fallback_note = ""

    def _new_model(self, device, scale_pos_weight):
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=self.n_estimators,
            max_depth=3,
            learning_rate=0.055,
            subsample=0.92,
            colsample_bytree=0.90,
            reg_lambda=3.0,
            random_state=self.seed,
            tree_method="hist",
            device=device,
            n_jobs=max(1, min(8, os.cpu_count() or 1)),
            scale_pos_weight=scale_pos_weight
        )

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        pos = max(1, int(y.sum()))
        neg = max(1, int(len(y) - y.sum()))
        spw = neg / pos
        try:
            self.model = self._new_model(self.device, spw)
            self.model.fit(Z, y, verbose=False)
            self.used_device = self.device
            return self
        except Exception as e:
            if self.device == "cuda":
                warnings.warn(f"XGBoost CUDA classifier failed; falling back to CPU: {e}")
                self.model = self._new_model("cpu", spw)
                self.model.fit(Z, y, verbose=False)
                self.used_device = "cpu"
                self.fallback_note = repr(e)[:500]
                return self
            raise

    def proba(self, X):
        Z = (X - self.mu) / self.sd
        return self.model.predict_proba(Z)[:, 1]


def cv_prediction(samples: Sequence[dict], acceleration: bool = False, noise: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    out = []
    for s in samples:
        h = s["hist_xy"].copy().astype(float)
        if noise > 0:
            h += rng.normal(0, noise, h.shape)
        v1 = (h[-1] - h[-6]) / (5*DT)
        if acceleration:
            v0 = (h[-6] - h[-11]) / (5*DT)
            acc = (v1 - v0) / (5*DT)
            pred = np.array([h[-1] + v1*(k+1)*DT + 0.5*acc*((k+1)*DT)**2 for k in range(PRED)])
        else:
            pred = np.array([h[-1] + v1*(k+1)*DT for k in range(PRED)])
        out.append(pred.ravel())
    return np.array(out)


def cv_lane_change_score(samples: Sequence[dict]):
    scores = []
    for s in samples:
        h = s["hist_xy"]
        vx = (h[-1, 0] - h[-8, 0]) / (7*DT)
        x_future = h[-1, 0] + vx * (PRED*DT)
        curr_lane = int(s["hist_lane"][-1])
        pred_lane = int(np.clip(np.round((x_future - LANE_W/2) / LANE_W), 0, 8))
        # Continuous score; strong lateral drift maps to high probability.
        drift = abs(x_future - h[-1, 0]) / LANE_W
        score = min(0.99, max(0.01, 0.15 + 0.7*drift + (0.25 if pred_lane != curr_lane else 0.0)))
        scores.append(score)
    return np.asarray(scores, dtype=float)


def metric_dict(Y, P):
    yt = Y.reshape(-1, PRED, 2)
    yp = P.reshape(-1, PRED, 2)
    d = np.sqrt(((yt - yp)**2).sum(axis=2))
    return {
        "ADE": float(d.mean()),
        "FDE": float(d[:, -1].mean()),
        "RMSE_x": float(np.sqrt(np.mean((yt[:, :, 0]-yp[:, :, 0])**2))),
        "RMSE_y": float(np.sqrt(np.mean((yt[:, :, 1]-yp[:, :, 1])**2))),
        "RMSE": float(np.sqrt(np.mean((yt-yp)**2))),
        "Lat_MAE": float(np.abs(yt[:, :, 0]-yp[:, :, 0]).mean()),
        "Lon_MAE": float(np.abs(yt[:, :, 1]-yp[:, :, 1]).mean()),
    }



def select_prediction_blend(Yval, Pbase, Pfull):
    """Select alpha on validation: P = (1-alpha)*Pbase + alpha*Pfull."""
    best_alpha, best_ade = 0.0, float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        P = (1-alpha)*Pbase + alpha*Pfull
        ade = metric_dict(Yval, P)["ADE"]
        if ade < best_ade:
            best_ade = ade
            best_alpha = float(alpha)
    return best_alpha, best_ade


def select_probability_blend(yval, pbase, pfull):
    """Select probability-level nonlinear fusion on validation F1."""
    best = (-1.0, 0.0, 0.5)
    for alpha in np.linspace(0.0, 1.0, 21):
        p = (1-alpha)*pbase + alpha*pfull
        thr = best_threshold(yval, p)
        f1 = binary_metrics(yval, p, thr)["F1"]
        if f1 > best[0]:
            best = (f1, float(alpha), float(thr))
    return best[1], best[2], best[0]

def binary_metrics(y, prob, threshold: float = 0.5):
    yh = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (yh == 0)).sum())
    fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum())
    tp = int(((y == 1) & (yh == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    acc = (tp + tn) / max(1, len(y))
    return {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1, "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def best_threshold(y, p):
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [(binary_metrics(y, p, t)["F1"], t) for t in thresholds]
    scores.sort(reverse=True)
    return float(scores[0][1])


def roc_curve(y, p):
    order = np.argsort(-p)
    y = y[order]
    Pn = max(1, int(y.sum()))
    Nn = max(1, int(len(y) - y.sum()))
    tp = fp = 0
    fpr = [0.0]; tpr = [0.0]
    for yy in y:
        if yy:
            tp += 1
        else:
            fp += 1
        fpr.append(fp/Nn); tpr.append(tp/Pn)
    area_fn = getattr(np, "trapezoid", np.trapz)
    return np.asarray(fpr), np.asarray(tpr), float(area_fn(tpr, fpr))


# -----------------------------------------------------------------------------
# Tables and figures
# -----------------------------------------------------------------------------
def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%")


def write_latex_tables(out: Path, meta: dict):
    table_dir = ensure(out/"tables")
    pred = pd.read_csv(out/"results"/"prediction_metrics.csv")
    with open(table_dir/"table_prediction_metrics.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("方法 & ADE/m & FDE/m & RMSE-x/m & RMSE-y/m & Lane-change F1 \\\\ \n\\midrule\n")
        for _, r in pred.iterrows():
            f.write(f"{tex_escape(r['Model'])} & {r['ADE']:.3f} & {r['FDE']:.3f} & {r['RMSE_x']:.3f} & {r['RMSE_y']:.3f} & {r['LaneChange_F1']:.3f} \\\\ \n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    ab = pd.read_csv(out/"results"/"ablation.csv")
    with open(table_dir/"table_ablation.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabularx}{\\textwidth}{lccX}\n\\toprule\n")
        f.write("特征组合 & ADE/m & FDE/m & 说明 \\\\ \n\\midrule\n")
        for _, r in ab.iterrows():
            f.write(f"{tex_escape(r['Feature_Set'])} & {r['ADE']:.3f} & {r['FDE']:.3f} & {tex_escape(r['Description'])} \\\\ \n")
        f.write("\\bottomrule\n\\end{tabularx}\n")

    with open(table_dir/"table_dataset.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabularx}{\\textwidth}{lX}\n\\toprule\n")
        rows = [
            ("公开数据源", "USDOT ITS DataHub / data.transportation.gov: Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data, DOI: 10.21949/1504477"),
            ("默认实验数据", f"NGSIM 字段兼容非线性基准；{meta['n_vehicles']} 辆车，{meta['n_rows']} 条记录，{meta['n_samples']} 个监督样本"),
            ("采样频率", "10 Hz；历史窗口 3.0 s，预测时域 5.0 s"),
            ("字段", "Vehicle_ID, Frame_ID, Global_Time, Local_X, Local_Y, v_Vel, v_Acc, Lane_ID, Preceding, Following, Space_Headway"),
            ("数据划分", f"按 Vehicle_ID 划分：训练 {meta['n_train']}，验证 {meta['n_val']}，测试 {meta['n_test']} 个样本"),
            ("真实数据接口", "code/download_ngsim_sample.py 可从 Socrata API 下载部分 NGSIM；code/run_experiments.py --input-csv 可替换为真实 CSV 并自动复现图表")
        ]
        for a, b in rows:
            f.write(f"{a} & {tex_escape(b)} \\\\ \n")
        f.write("\\bottomrule\n\\end{tabularx}\n")

    cls = pd.read_csv(out/"results"/"classification_metrics.csv")
    with open(table_dir/"table_classification.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("方法 & 阈值 & Accuracy & Precision & Recall & F1 & AUC \\\\ \n\\midrule\n")
        for _, r in cls.iterrows():
            f.write(f"{tex_escape(r['Model'])} & {r['Threshold']:.2f} & {r['Accuracy']:.3f} & {r['Precision']:.3f} & {r['Recall']:.3f} & {r['F1']:.3f} & {r['AUC']:.3f} \\\\ \n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def save_figures(out: Path, df: pd.DataFrame, samples, test, Yte, predictions, classification_prob, classification_thresholds, Cte, full_importance):
    fig_dir = ensure(out/"figures")
    res_dir = out/"results"
    dpi = 300

    # Framework figure
    plt.figure(figsize=(5.5, 3.25))
    ax = plt.gca(); ax.axis("off")
    boxes = [
        (0.02, 0.62, 0.22, 0.18, "History window\n3 s trajectory"),
        (0.31, 0.76, 0.22, 0.16, "CV prior\nkinematic baseline"),
        (0.31, 0.52, 0.22, 0.16, "Delay embedding\nphase space"),
        (0.31, 0.28, 0.22, 0.16, "RQA / Lyapunov\nSampEn / DFA"),
        (0.60, 0.66, 0.22, 0.16, "Graph interaction\ndensity / headway"),
        (0.60, 0.40, 0.22, 0.16, "Feature fusion\nstandardization"),
        (0.84, 0.55, 0.14, 0.18, "XGBoost\nresidual"),
        (0.84, 0.25, 0.14, 0.16, "Lane-change\nclassifier"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.0))
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=8)
    arrows = [((0.24,0.71),(0.31,0.84)), ((0.24,0.71),(0.31,0.60)), ((0.24,0.71),(0.31,0.36)),
              ((0.53,0.84),(0.60,0.74)), ((0.53,0.60),(0.60,0.48)), ((0.53,0.36),(0.60,0.48)),
              ((0.82,0.48),(0.84,0.64)), ((0.82,0.48),(0.84,0.33))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(0.53, 0.08, r"$\hat{Y}=Y_{CV}+f_{XGB}(\phi_{hist},\phi_{NL},\phi_{graph})$", fontsize=9, ha="center")
    plt.tight_layout()
    plt.savefig(fig_dir/"fig00_framework.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # Trajectories
    plt.figure(figsize=(4.2, 3.2))
    for vid, g in df.groupby("Vehicle_ID"):
        if int(vid) <= 16:
            plt.plot(g.Local_Y, g.Local_X, linewidth=0.8, alpha=0.85)
    plt.xlabel("Longitudinal position y (m)")
    plt.ylabel("Lateral position x (m)")
    plt.title("Trajectory samples")
    plt.tight_layout()
    plt.savefig(fig_dir/"fig01_trajectory_samples.png", dpi=dpi)
    plt.close()

    # Phase space and recurrence plot combined side-by-side in one image.
    # A 2-D projection is used here for fast, stable PDF reproduction.
    s0 = test[0]
    yh = s0["hist_xy"][:, 1]
    E = _embed(yh, m=3, tau=4)
    R = recurrence_matrix(yh, m=3, tau=4)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    axes[0].plot(E[:, 0], E[:, 1], marker="o", markersize=2, linewidth=0.8)
    axes[0].set_xlabel("y(t)"); axes[0].set_ylabel("y(t-tau)")
    axes[0].set_title("Phase-space projection", fontsize=8)
    axes[1].imshow(R, origin="lower", interpolation="nearest")
    axes[1].set_title("Recurrence plot", fontsize=8)
    axes[1].set_xlabel("State index"); axes[1].set_ylabel("State index")
    fig.tight_layout()
    fig.savefig(fig_dir/"fig02_phase_recurrence.png", dpi=dpi)
    plt.close(fig)

    # Nonlinear feature distributions
    vals = []
    for ss in test[:160]:
        h = ss["hist_xy"]
        xh, yh = h[:, 0], h[:, 1]
        vx = np.gradient(xh, DT); vy = np.gradient(yh, DT)
        speed = np.sqrt(vx*vx + vy*vy)
        rr, det, _ = rqa_metrics(yh)
        vals.append([sample_entropy(speed), np.nanmean(multiscale_sampen(speed)), dfa_alpha(speed), lyapunov_proxy(yh), rr, det])
    vals = np.asarray(vals)
    labels = ["SampEn", "MSE", "DFA", "Lyap", "RR", "DET"]
    means = np.nanmean(vals, axis=0)
    stds = np.nanstd(vals, axis=0)
    plt.figure(figsize=(5.2, 3.0))
    xpos = np.arange(len(labels))
    plt.bar(xpos, means, yerr=stds, capsize=2)
    plt.xticks(xpos, labels, rotation=20)
    plt.ylabel("Mean +/- std")
    plt.title("Nonlinear feature distributions")
    plt.tight_layout()
    plt.savefig(fig_dir/"fig03_nonlinear_feature_distribution.png", dpi=dpi)
    plt.close()

    # Main metrics
    metrics = pd.read_csv(res_dir/"prediction_metrics.csv")
    xloc = np.arange(len(metrics)); width = 0.36
    plt.figure(figsize=(5.2, 3.0))
    plt.bar(xloc-width/2, metrics.ADE, width, label="ADE")
    plt.bar(xloc+width/2, metrics.FDE, width, label="FDE")
    plt.xticks(xloc, metrics.Model, rotation=18, ha="right")
    plt.ylabel("Error (m)"); plt.title("Prediction error comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir/"fig04_prediction_metrics.png", dpi=dpi)
    plt.close()

    # Horizon errors
    hdf = pd.read_csv(res_dir/"horizon_errors.csv")
    plt.figure(figsize=(5.2, 3.2))
    for name, g in hdf.groupby("Model"):
        plt.plot(g.Horizon_s, g.Error_m, marker="o", linewidth=1.1, label=name)
    plt.xlabel("Prediction horizon (s)"); plt.ylabel("Displacement error (m)")
    plt.title("Error growth across horizons")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(fig_dir/"fig05_horizon_error.png", dpi=dpi)
    plt.close()

    # Ablation
    ab = pd.read_csv(res_dir/"ablation.csv")
    plt.figure(figsize=(5.2, 3.1))
    plt.bar(ab.Feature_Set, ab.ADE)
    plt.xticks(rotation=18, ha="right")
    plt.ylabel("ADE (m)"); plt.title("Feature ablation")
    plt.tight_layout()
    plt.savefig(fig_dir/"fig06_ablation.png", dpi=dpi)
    plt.close()

    # Embedding sensitivity
    ed = pd.read_csv(res_dir/"sensitivity_embedding.csv")
    plt.figure(figsize=(4.4, 3.0))
    plt.plot(ed.Embedding_dim, ed.ADE, marker="o", label="ADE")
    plt.plot(ed.Embedding_dim, ed.FDE, marker="s", label="FDE")
    plt.xlabel("Embedding dimension m"); plt.ylabel("Error (m)")
    plt.title("Embedding-dimension sensitivity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir/"fig07_embedding_sensitivity.png", dpi=dpi)
    plt.close()

    # Multiscale SampEn by behavior
    mse = pd.read_csv(res_dir/"mse_by_behavior.csv")
    plt.figure(figsize=(4.8, 3.2))
    for behavior, g in mse.groupby("Behavior"):
        plt.plot(g.Scale, g.SampEn, marker="o", linewidth=1.1, label=behavior)
    plt.xlabel("Scale"); plt.ylabel("SampEn")
    plt.title("Multiscale complexity by behavior")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir/"fig08_multiscale_complexity.png", dpi=dpi)
    plt.close()

    # Noise robustness
    nr = pd.read_csv(res_dir/"noise_robustness.csv")
    plt.figure(figsize=(4.9, 3.1))
    for model, g in nr.groupby("Model"):
        plt.plot(g.NoisePercent, g.ADE, marker="o", linewidth=1.1, label=model)
    plt.xlabel("Observation noise (% of lane width)"); plt.ylabel("ADE (m)")
    plt.title("Noise robustness")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir/"fig09_noise_robustness.png", dpi=dpi)
    plt.close()

    # Classification ROC and confusion for XGB nonlinear
    prob = classification_prob["XGBoost+nonlinear"]
    thr = classification_thresholds["XGBoost+nonlinear"]
    fpr, tpr, auc = roc_curve(Cte, prob)
    bm = binary_metrics(Cte, prob, thr)
    cm = np.array([[bm["TN"], bm["FP"]], [bm["FN"], bm["TP"]]])
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7))
    axes[0].imshow(cm)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Stay", "Change"], fontsize=7)
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Stay", "Change"], fontsize=7)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title("Confusion matrix", fontsize=8)
    for (i, j), val in np.ndenumerate(cm):
        axes[0].text(j, i, str(val), ha="center", va="center", fontsize=8)
    axes[1].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC", fontsize=8); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir/"fig10_lane_change.png", dpi=dpi)
    plt.close(fig)

    # Feature importance
    top = full_importance.head(14).iloc[::-1]
    plt.figure(figsize=(5.2, 3.6))
    plt.barh(top.Feature, top.Importance)
    plt.xlabel("Importance")
    plt.title("XGBoost nonlinear feature importance")
    plt.tight_layout()
    plt.savefig(fig_dir/"fig11_feature_importance.png", dpi=dpi)
    plt.close()

    # Qualitative predictions
    yt = Yte.reshape(-1, PRED, 2)
    plt.figure(figsize=(5.2, 3.1))
    cases = [0, min(10, len(test)-1), min(25, len(test)-1)]
    for idx, k in enumerate(cases):
        h = test[k]["hist_xy"]
        plt.plot(h[:, 1], h[:, 0], linestyle=":", linewidth=0.9)
        plt.plot(yt[k, :, 1], yt[k, :, 0], linewidth=1.8, label="Ground truth" if idx == 0 else None)
        for label, lw in [("XGBoost+nonlinear", 1.4), ("CV", 0.9)]:
            P = predictions[label].reshape(-1, PRED, 2)
            plt.plot(P[k, :, 1], P[k, :, 0], linewidth=lw, label=label if idx == 0 else None)
    plt.xlabel("Longitudinal position y (m)"); plt.ylabel("Lateral position x (m)")
    plt.title("Qualitative trajectory cases")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir/"fig12_qualitative_cases.png", dpi=dpi)
    plt.close()


def behavior_label(sample: dict):
    future_change = int(np.any(sample["future_lane"] != sample["hist_lane"][-1]))
    h = sample["hist_xy"]
    vx = np.gradient(h[:, 0], DT); vy = np.gradient(h[:, 1], DT)
    ay = np.gradient(vy, DT)
    if future_change:
        return "Lane-change"
    if np.min(ay) < -2.0:
        return "Hard-brake"
    if sample["graph"][1] < 24.0:
        return "Following"
    return "Straight"


def make_mse_by_behavior(samples, out_csv: Path):
    rows = []
    for behavior in ["Straight", "Following", "Lane-change", "Hard-brake"]:
        chosen = [s for s in samples if behavior_label(s) == behavior]
        if not chosen:
            continue
        vals = []
        for s in chosen[:120]:
            h = s["hist_xy"]
            vx = np.gradient(h[:, 0], DT); vy = np.gradient(h[:, 1], DT)
            speed = np.sqrt(vx*vx + vy*vy)
            vals.append(multiscale_sampen(speed, scales=(1, 2, 3, 4, 5)))
        arr = np.vstack(vals)
        for idx, scale in enumerate([1, 2, 3, 4, 5]):
            rows.append({"Behavior": behavior, "Scale": scale, "SampEn": float(np.nanmean(arr[:, idx]))})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def run(args):
    out = Path(args.output)
    ensure(out/"results"); ensure(out/"figures"); ensure(out/"tables"); ensure(out/"data"/"generated"); ensure(out/"data"/"raw")
    gpu_status = detect_acceleration(args.device)
    (out/"results"/"gpu_status.json").write_text(json.dumps(gpu_status, ensure_ascii=False, indent=2), encoding="utf-8")
    device = gpu_status["device_selected"]

    if args.input_csv:
        df = load_real_ngsim_csv(args.input_csv, input_units=args.input_units)
        vehicle_ids, x, y, v, a, lane = matrix_from_df(df, max_frames=args.max_frames)
        source = "real_or_external_ngsim_csv"
    else:
        x, y, v, a, lane = simulate_ngsim_compatible(seed=args.seed, n=args.n_vehicles, T=args.n_frames, lanes=args.n_lanes)
        vehicle_ids = np.arange(1, x.shape[0]+1)
        df = dataframe_from_matrices(x, y, v, a, lane)
        source = "ngsim_field_compatible_nonlinear_benchmark"
    df.to_csv(out/"data"/"generated"/"ngsim_compatible_benchmark.csv", index=False)

    samples = make_samples(vehicle_ids, x, y, v, a, lane, stride=args.stride)
    train, val, test = split_samples(samples, seed=args.seed)
    Ytr, Ctr = targets(train); Yva, Cva = targets(val); Yte, Cte = targets(test)

    groups_base = ("base", "poly")
    groups_full = ("base", "delay", "lyap", "rqa", "entropy", "graph", "poly")

    Xtr_base, base_names = feature_matrix(train, groups_base, return_names=True, seed=args.seed)
    Xva_base = feature_matrix(val, groups_base, seed=args.seed)
    Xte_base = feature_matrix(test, groups_base, seed=args.seed)
    Xtr_full, full_names = feature_matrix(train, groups_full, m=3, return_names=True, seed=args.seed)
    Xva_full = feature_matrix(val, groups_full, m=3, seed=args.seed)
    Xte_full = feature_matrix(test, groups_full, m=3, seed=args.seed)
    groups_nl_only = ("delay", "lyap", "rqa", "entropy", "graph", "poly")
    Xtr_nl, nl_names = feature_matrix(train, groups_nl_only, m=3, return_names=True, seed=args.seed)
    Xva_nl = feature_matrix(val, groups_nl_only, m=3, seed=args.seed)
    Xte_nl = feature_matrix(test, groups_nl_only, m=3, seed=args.seed)

    P_cv_tr = cv_prediction(train)
    P_cv_va = cv_prediction(val)
    P_cv_te = cv_prediction(test)
    residual_tr = Ytr - P_cv_tr
    residual_basis_tr = residual_to_basis(residual_tr)

    P_ca_te = cv_prediction(test, acceleration=True)
    predictions = {"CV": P_cv_te, "CA": P_ca_te}

    ridge_base = StandardRidge(lam=8.0).fit(Xtr_base, residual_tr)
    P_ridge_va_base = P_cv_va + ridge_base.predict(Xva_base)
    P_ridge_te_base = P_cv_te + ridge_base.predict(Xte_base)
    predictions["Ridge"] = P_ridge_te_base

    ridge_nl_only = StandardRidge(lam=18.0).fit(Xtr_nl, residual_tr)
    predictions["Ridge-nonlinear-only"] = P_cv_te + ridge_nl_only.predict(Xte_nl)

    ridge_full = StandardRidge(lam=18.0).fit(Xtr_full, residual_tr)
    P_ridge_va_full = P_cv_va + ridge_full.predict(Xva_full)
    P_ridge_te_full_raw = P_cv_te + ridge_full.predict(Xte_full)
    alpha_ridge, _ = select_prediction_blend(Yva, P_ridge_va_base, P_ridge_va_full)
    predictions["Ridge+nonlinear"] = (1-alpha_ridge)*P_ridge_te_base + alpha_ridge*P_ridge_te_full_raw

    xgb_base = StandardXGBRegressor(seed=args.seed, device=device, n_estimators=args.xgb_estimators).fit(Xtr_base, residual_basis_tr)
    P_xgb_va_base = P_cv_va + basis_to_residual(xgb_base.predict(Xva_base))
    P_xgb_te_base = P_cv_te + basis_to_residual(xgb_base.predict(Xte_base))
    predictions["XGBoost"] = P_xgb_te_base

    xgb_nl_only = StandardXGBRegressor(seed=args.seed, device=device, n_estimators=args.xgb_estimators).fit(Xtr_nl, residual_basis_tr)
    predictions["XGBoost-nonlinear-only"] = P_cv_te + basis_to_residual(xgb_nl_only.predict(Xte_nl))

    xgb_candidates = [
        ("delay", ("base", "delay", "poly")),
        ("lyap", ("base", "lyap", "poly")),
        ("entropy", ("base", "entropy", "poly")),
        ("graph", ("base", "graph", "poly")),
        ("lyap+graph", ("base", "lyap", "graph", "poly")),
        ("entropy+graph", ("base", "entropy", "graph", "poly")),
        ("full", groups_full),
    ]
    best_xgb = {
        "name": "base",
        "model": xgb_base,
        "names": base_names,
        "Xte": Xte_base,
        "Pva": P_xgb_va_base,
        "Pte_raw": P_xgb_te_base,
        "alpha": 0.0,
        "val_ade": metric_dict(Yva, P_xgb_va_base)["ADE"],
    }
    xgb_candidate_rows = []
    for cand_name, cand_groups in xgb_candidates:
        Xtr_c, names_c = feature_matrix(train, cand_groups, m=3, return_names=True, seed=args.seed)
        Xva_c = feature_matrix(val, cand_groups, m=3, seed=args.seed)
        Xte_c = feature_matrix(test, cand_groups, m=3, seed=args.seed)
        model_c = StandardXGBRegressor(seed=args.seed, device=device, n_estimators=args.xgb_estimators).fit(Xtr_c, residual_basis_tr)
        Pva_c = P_cv_va + basis_to_residual(model_c.predict(Xva_c))
        Pte_c = P_cv_te + basis_to_residual(model_c.predict(Xte_c))
        alpha_c, val_ade_c = select_prediction_blend(Yva, P_xgb_va_base, Pva_c)
        val_fde_c = metric_dict(Yva, (1-alpha_c)*P_xgb_va_base + alpha_c*Pva_c)["FDE"]
        xgb_candidate_rows.append({"Candidate": cand_name, "Alpha": alpha_c, "Val_ADE": val_ade_c, "Val_FDE": val_fde_c})
        if val_ade_c < best_xgb["val_ade"]:
            best_xgb = {
                "name": cand_name,
                "groups": cand_groups,
                "model": model_c,
                "names": names_c,
                "Xte": Xte_c,
                "Pva": Pva_c,
                "Pte_raw": Pte_c,
                "alpha": alpha_c,
                "val_ade": val_ade_c,
            }
    pd.DataFrame(xgb_candidate_rows).to_csv(out/"results"/"xgboost_nonlinear_candidates.csv", index=False)
    xgb_full = best_xgb["model"]
    groups_xgb_selected = best_xgb.get("groups", groups_base)
    full_names = best_xgb["names"]
    Xte_full_for_importance = best_xgb["Xte"]
    alpha_xgb = float(best_xgb["alpha"])
    P_xgb_te_full_raw = best_xgb["Pte_raw"]
    predictions["XGBoost+nonlinear"] = (1-alpha_xgb)*P_xgb_te_base + alpha_xgb*P_xgb_te_full_raw

    # Classification branch with validation-threshold selection.
    prob_val = {}; prob_test = {}; thresholds = {}
    prob_val["CV"] = cv_lane_change_score(val); prob_test["CV"] = cv_lane_change_score(test)
    thresholds["CV"] = best_threshold(Cva, prob_val["CV"])
    prob_val["CA"] = prob_val["CV"]; prob_test["CA"] = prob_test["CV"]; thresholds["CA"] = thresholds["CV"]

    raw_prob_val = {}
    raw_prob_test = {}
    for label, Xtr, Xva, Xte, kind in [
        ("Ridge", Xtr_base, Xva_base, Xte_base, "logistic"),
        ("Ridge-nonlinear-only", Xtr_nl, Xva_nl, Xte_nl, "logistic"),
        ("Ridge_full_raw", Xtr_full, Xva_full, Xte_full, "logistic"),
        ("XGBoost", Xtr_base, Xva_base, Xte_base, "xgb"),
        ("XGBoost-nonlinear-only", Xtr_nl, Xva_nl, Xte_nl, "xgb"),
        ("XGBoost_full_raw", Xtr_full, Xva_full, Xte_full, "xgb"),
    ]:
        if kind == "logistic":
            clf = WeightedLogistic().fit(Xtr, Ctr)
        else:
            clf = StandardXGBClassifier(seed=args.seed, device=device, n_estimators=args.xgb_estimators).fit(Xtr, Ctr)
        raw_prob_val[label] = clf.proba(Xva)
        raw_prob_test[label] = clf.proba(Xte)

    prob_val["Ridge"] = raw_prob_val["Ridge"]
    prob_test["Ridge"] = raw_prob_test["Ridge"]
    thresholds["Ridge"] = best_threshold(Cva, prob_val["Ridge"])
    prob_val["Ridge-nonlinear-only"] = raw_prob_val["Ridge-nonlinear-only"]
    prob_test["Ridge-nonlinear-only"] = raw_prob_test["Ridge-nonlinear-only"]
    thresholds["Ridge-nonlinear-only"] = best_threshold(Cva, prob_val["Ridge-nonlinear-only"])
    a_prob_ridge, thr_ridge_nl, _ = select_probability_blend(Cva, raw_prob_val["Ridge"], raw_prob_val["Ridge_full_raw"])
    prob_val["Ridge+nonlinear"] = (1-a_prob_ridge)*raw_prob_val["Ridge"] + a_prob_ridge*raw_prob_val["Ridge_full_raw"]
    prob_test["Ridge+nonlinear"] = (1-a_prob_ridge)*raw_prob_test["Ridge"] + a_prob_ridge*raw_prob_test["Ridge_full_raw"]
    thresholds["Ridge+nonlinear"] = thr_ridge_nl

    prob_val["XGBoost"] = raw_prob_val["XGBoost"]
    prob_test["XGBoost"] = raw_prob_test["XGBoost"]
    thresholds["XGBoost"] = best_threshold(Cva, prob_val["XGBoost"])
    prob_val["XGBoost-nonlinear-only"] = raw_prob_val["XGBoost-nonlinear-only"]
    prob_test["XGBoost-nonlinear-only"] = raw_prob_test["XGBoost-nonlinear-only"]
    thresholds["XGBoost-nonlinear-only"] = best_threshold(Cva, prob_val["XGBoost-nonlinear-only"])
    a_prob_xgb, thr_xgb_nl, _ = select_probability_blend(Cva, raw_prob_val["XGBoost"], raw_prob_val["XGBoost_full_raw"])
    prob_val["XGBoost+nonlinear"] = (1-a_prob_xgb)*raw_prob_val["XGBoost"] + a_prob_xgb*raw_prob_val["XGBoost_full_raw"]
    prob_test["XGBoost+nonlinear"] = (1-a_prob_xgb)*raw_prob_test["XGBoost"] + a_prob_xgb*raw_prob_test["XGBoost_full_raw"]
    thresholds["XGBoost+nonlinear"] = thr_xgb_nl

    # Main metrics
    rows = []
    cls_rows = []
    metric_order = ["CV", "CA", "Ridge", "Ridge-nonlinear-only", "Ridge+nonlinear", "XGBoost", "XGBoost-nonlinear-only", "XGBoost+nonlinear"]
    for label in metric_order:
        m = metric_dict(Yte, predictions[label])
        cls = binary_metrics(Cte, prob_test[label], thresholds[label])
        fpr, tpr, auc = roc_curve(Cte, prob_test[label])
        m["Model"] = label
        m["LaneChange_F1"] = cls["F1"]
        rows.append(m)
        cls_row = {"Model": label, "Threshold": thresholds[label], "AUC": auc, **cls}
        cls_rows.append(cls_row)
    pd.DataFrame(rows).to_csv(out/"results"/"prediction_metrics.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(out/"results"/"classification_metrics.csv", index=False)
    np.savez_compressed(out/"results"/"predictions_test.npz",
        Yte=Yte, Cte=Cte,
        CV=predictions["CV"], CA=predictions["CA"], Ridge=predictions["Ridge"],
        Ridge_nonlinear_only=predictions["Ridge-nonlinear-only"],
        Ridge_nonlinear=predictions["Ridge+nonlinear"],
        XGBoost=predictions["XGBoost"],
        XGBoost_nonlinear_only=predictions["XGBoost-nonlinear-only"],
        XGBoost_nonlinear=predictions["XGBoost+nonlinear"])
    print("[stage] main metrics done", flush=True)

    # Horizon errors
    hrows = []
    yt = Yte.reshape(-1, PRED, 2)
    for label, P in predictions.items():
        yp = P.reshape(-1, PRED, 2)
        d = np.sqrt(((yt-yp)**2).sum(axis=2))
        for h in [10, 20, 30, 40, 50]:
            hrows.append({"Model": label, "Horizon_s": h*DT, "Error_m": float(d[:, h-1].mean())})
    pd.DataFrame(hrows).to_csv(out/"results"/"horizon_errors.csv", index=False)
    print("[stage] horizon errors done", flush=True)

    # Ablation with same Ridge residual learner to isolate feature contribution.
    ablation_specs = [
        ("Kinematics", ("base", "poly"), "基础运动学输入"),
        ("+ delay embedding", ("base", "delay", "poly"), "加入历史状态的延迟嵌入结构"),
        ("+ Lyapunov proxy", ("base", "delay", "lyap", "poly"), "加入有限窗口扰动发散与 jerk 统计"),
        ("+ RQA", ("base", "delay", "lyap", "rqa", "poly"), "加入递归率、确定性和对角线长度"),
        ("+ entropy/DFA", ("base", "delay", "lyap", "rqa", "entropy", "poly"), "加入样本熵、多尺度样本熵和 DFA/Hurst 代理"),
        ("+ graph interaction", groups_full, "加入邻域图密度、车头间距和相对速度"),
    ]
    ab_rows = []
    for name, groups, desc in ablation_specs:
        Xt = feature_matrix(train, groups, seed=args.seed)
        Xs = feature_matrix(test, groups, seed=args.seed)
        pred = P_cv_te + StandardRidge(lam=8.0).fit(Xt, residual_tr).predict(Xs)
        mm = metric_dict(Yte, pred)
        ab_rows.append({"Feature_Set": name, "ADE": mm["ADE"], "FDE": mm["FDE"], "Description": desc})
    pd.DataFrame(ab_rows).to_csv(out/"results"/"ablation.csv", index=False)
    print("[stage] ablation done", flush=True)

    # Embedding dimension sensitivity m=2..7. Use a validation-sized subset to
    # keep the report reproducible on laptops while preserving the parameter trend.
    sens_train = train[:min(len(train), 60)]
    sens_test = test[:min(len(test), 24)]
    Ys_tr, _ = targets(sens_train); Ys_te, _ = targets(sens_test)
    Pcv_s_tr = cv_prediction(sens_train); Pcv_s_te = cv_prediction(sens_test)
    R_s_tr = Ys_tr - Pcv_s_tr
    sens_groups = ("base", "delay", "poly")
    emb_rows = []
    for mm_dim in [2, 3, 4, 5, 6, 7]:
        Xt = feature_matrix(sens_train, sens_groups, m=mm_dim, seed=args.seed)
        Xs = feature_matrix(sens_test, sens_groups, m=mm_dim, seed=args.seed)
        pred = Pcv_s_te + StandardRidge(lam=12.0).fit(Xt, R_s_tr).predict(Xs)
        md = metric_dict(Ys_te, pred)
        emb_rows.append({"Embedding_dim": mm_dim, "ADE": md["ADE"], "FDE": md["FDE"]})
    pd.DataFrame(emb_rows).to_csv(out/"results"/"sensitivity_embedding.csv", index=False)
    print("[stage] embedding sensitivity done", flush=True)

    # Two-dimensional m-tau sensitivity on the same compact split.
    mtau_rows = []
    for mm_dim in [2, 3, 4, 5, 6, 7]:
        for tau_val in [1, 2, 3, 4, 5]:
            Xt = feature_matrix(sens_train, sens_groups, m=mm_dim, tau=tau_val, seed=args.seed)
            Xs = feature_matrix(sens_test, sens_groups, m=mm_dim, tau=tau_val, seed=args.seed)
            pred = Pcv_s_te + StandardRidge(lam=12.0).fit(Xt, R_s_tr).predict(Xs)
            md = metric_dict(Ys_te, pred)
            mtau_rows.append({"Embedding_dim": mm_dim, "Tau": tau_val, "ADE": md["ADE"], "FDE": md["FDE"]})
    pd.DataFrame(mtau_rows).to_csv(out/"results"/"sensitivity_m_tau.csv", index=False)
    print("[stage] m-tau sensitivity done", flush=True)

    # SampEn r and RQA threshold sensitivity. These retrain a compact Ridge
    # model with the full feature family so the parameter effect is reflected
    # in ADE/FDE, not only in descriptive statistics.
    global SAMPEN_R_RATIO, RQA_EPS_PERCENTILE
    old_r, old_eps = SAMPEN_R_RATIO, RQA_EPS_PERCENTILE
    samp_rows = []
    for ratio in [0.10, 0.15, 0.20, 0.25, 0.30]:
        SAMPEN_R_RATIO = ratio
        Xt = feature_matrix(sens_train, groups_full, m=3, seed=args.seed)
        Xs = feature_matrix(sens_test, groups_full, m=3, seed=args.seed)
        pred = Pcv_s_te + StandardRidge(lam=18.0).fit(Xt, R_s_tr).predict(Xs)
        md = metric_dict(Ys_te, pred)
        samp_rows.append({"R_ratio": ratio, "ADE": md["ADE"], "FDE": md["FDE"]})
    pd.DataFrame(samp_rows).to_csv(out/"results"/"sensitivity_sampen_r.csv", index=False)

    rqa_rows = []
    for pct in [5, 10, 15, 20]:
        RQA_EPS_PERCENTILE = float(pct)
        Xt = feature_matrix(sens_train, groups_full, m=3, seed=args.seed)
        Xs = feature_matrix(sens_test, groups_full, m=3, seed=args.seed)
        pred = Pcv_s_te + StandardRidge(lam=18.0).fit(Xt, R_s_tr).predict(Xs)
        md = metric_dict(Ys_te, pred)
        rqa_rows.append({"RQA_eps_percentile": pct, "ADE": md["ADE"], "FDE": md["FDE"]})
    pd.DataFrame(rqa_rows).to_csv(out/"results"/"sensitivity_rqa_eps.csv", index=False)
    SAMPEN_R_RATIO, RQA_EPS_PERCENTILE = old_r, old_eps
    print("[stage] SampEn/RQA sensitivity done", flush=True)

    # Noise robustness: 0, 2, 5, 10 percent of lane width.
    noise_rows = []
    noise_test = test[:min(len(test), 48)]
    Yn_te, _ = targets(noise_test)
    for pct in [0, 2, 5, 10]:
        sigma = LANE_W * pct / 100.0
        Pcv_noise = cv_prediction(noise_test, noise=sigma, seed=args.seed+77)
        Xte_b_noise = feature_matrix(noise_test, groups_base, noise=sigma, seed=args.seed+77)
        Xte_f_noise = feature_matrix(noise_test, groups_xgb_selected, noise=sigma, seed=args.seed+77)
        pred_base = Pcv_noise + basis_to_residual(xgb_base.predict(Xte_b_noise))
        pred_full_raw_noise = Pcv_noise + basis_to_residual(xgb_full.predict(Xte_f_noise))
        pred_full = (1-alpha_xgb)*pred_base + alpha_xgb*pred_full_raw_noise
        for label, pred in [("XGBoost", pred_base), ("XGBoost+nonlinear", pred_full)]:
            md = metric_dict(Yn_te, pred)
            noise_rows.append({"NoisePercent": pct, "Noise_sigma_m": sigma, "Model": label, "ADE": md["ADE"], "FDE": md["FDE"], "N": len(noise_test)})
    pd.DataFrame(noise_rows).to_csv(out/"results"/"noise_robustness.csv", index=False)
    print("[stage] noise robustness done", flush=True)

    # Multiscale complexity by behavior.
    make_mse_by_behavior(test, out/"results"/"mse_by_behavior.csv")
    print("[stage] multiscale behavior done", flush=True)

    # Feature importance.
    imp = xgb_full.importance()
    if imp is None or len(imp) != len(full_names):
        imp = np.zeros(len(full_names))
    imp_df = pd.DataFrame({"Feature": full_names, "Importance": imp})
    imp_df = imp_df.sort_values("Importance", ascending=False)
    imp_df.to_csv(out/"results"/"feature_importance.csv", index=False)
    print("[stage] feature importance done", flush=True)

    # Permutation importance on a bounded test subset.
    perm_n = min(len(test), 1200)
    rng_perm = np.random.default_rng(args.seed + 202)
    base_subset = P_xgb_te_base[:perm_n]
    Y_perm = Yte[:perm_n]
    X_perm = Xte_full_for_importance[:perm_n].copy()
    orig_raw = P_cv_te[:perm_n] + basis_to_residual(xgb_full.predict(X_perm))
    orig_pred = (1-alpha_xgb)*base_subset + alpha_xgb*orig_raw
    orig_ade = metric_dict(Y_perm, orig_pred)["ADE"]
    perm_rows = []
    candidate_idx = np.argsort(-imp)[:min(30, len(full_names))]
    for j in candidate_idx:
        Xp = X_perm.copy()
        Xp[:, j] = rng_perm.permutation(Xp[:, j])
        raw = P_cv_te[:perm_n] + basis_to_residual(xgb_full.predict(Xp))
        pred = (1-alpha_xgb)*base_subset + alpha_xgb*raw
        ade = metric_dict(Y_perm, pred)["ADE"]
        perm_rows.append({"Feature": full_names[j], "ADE_perm": ade, "ADE_orig": orig_ade, "Importance": ade - orig_ade})
    pd.DataFrame(perm_rows).sort_values("Importance", ascending=False).to_csv(out/"results"/"permutation_importance.csv", index=False)
    print("[stage] permutation importance done", flush=True)

    # ROC points for selected model.
    fpr, tpr, auc = roc_curve(Cte, prob_test["XGBoost+nonlinear"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out/"results"/"roc_curve_points.csv", index=False)

    # Metadata.
    meta = {
        "data_source": source,
        "n_rows": int(len(df)),
        "n_vehicles": int(df.Vehicle_ID.nunique()),
        "n_samples": int(len(samples)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "history_seconds": HIST*DT,
        "prediction_seconds": PRED*DT,
        "sampling_hz": 10,
        "split_strategy": "Vehicle_ID split when possible; otherwise sample split",
        "seed": args.seed,
        "xgboost_regressor_used_device": xgb_full.used_device,
        "xgboost_cuda_fallback_note": xgb_full.fallback_note,
        "xgboost_nonlinear_selected_candidate": best_xgb["name"],
        "ridge_nonlinear_validation_blend_alpha": alpha_ridge,
        "xgboost_nonlinear_validation_blend_alpha": alpha_xgb,
        "ridge_lane_probability_blend_alpha": a_prob_ridge,
        "xgboost_lane_probability_blend_alpha": a_prob_xgb,
        "ngsim_official_doi": "10.21949/1504477",
        "ngsim_official_api_metadata": "https://data.transportation.gov/api/views/8ect-6jqj"
    }
    (out/"results"/"experiment_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    write_latex_tables(out, meta)
    print("[stage] latex tables done", flush=True)
    if not getattr(args, "skip_figures", False):
        save_figures(out, df, samples, test, Yte, predictions, prob_test, thresholds, Cte, imp_df)
        print("[stage] figures done", flush=True)

    print(json.dumps({"meta": meta, "gpu_status": gpu_status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=None, help="Optional real NGSIM-like CSV.")
    parser.add_argument("--input-units", default="feet", choices=["feet", "meters"], help="Units of external CSV; official NGSIM is feet.")
    parser.add_argument("--output", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="GPU-first: auto uses CUDA if visible.")
    parser.add_argument("--xgb-estimators", type=int, default=20)
    parser.add_argument("--n-vehicles", type=int, default=24)
    parser.add_argument("--n-frames", type=int, default=400)
    parser.add_argument("--n-lanes", type=int, default=4)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--skip-figures", action="store_true", help="Generate tables/results only; useful for quick tuning.")
    run(parser.parse_args())
