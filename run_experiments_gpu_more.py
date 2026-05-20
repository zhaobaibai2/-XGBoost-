# -*- coding: utf-8 -*-
"""
GPU-accelerated wrapper for run_experiments.py.

This file does not overwrite the original script. It monkey-patches the slow
CPU-heavy parts with torch/CUDA implementations where practical:
  - cleaned track limiting
  - matrix construction by Track_ID / Vehicle_ID
  - neighbor graph calculation in make_samples
  - batched feature_matrix
  - Ridge / logistic / CV / metrics / residual basis

XGBoost remains handled by run_experiments.py with device='cuda'.
Matplotlib and CSV IO still run on CPU.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

import run_experiments as r

try:
    import torch
except Exception as e:
    torch = None
    print("[warn] torch not importable:", repr(e))


MAX_TRACKS = 0
SEED = 42
GPU_BATCH = 4096
NEIGHBOR_BATCH = 2048
DEVICE_NAME = "cuda"

_ORIG_LOAD = r.load_real_ngsim_csv
_ORIG_MAKE_SAMPLES = r.make_samples


def dev():
    if torch is None:
        return None
    if DEVICE_NAME == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_real_ngsim_csv_limited(path, input_units="feet"):
    df = _ORIG_LOAD(path, input_units=input_units)

    group_col = "Track_ID" if "Track_ID" in df.columns else "Vehicle_ID"

    if MAX_TRACKS and MAX_TRACKS > 0:
        keys = pd.Series(df[group_col].dropna().unique())
        if len(keys) > MAX_TRACKS:
            rng = np.random.default_rng(SEED)
            chosen = set(rng.choice(keys.to_numpy(), size=MAX_TRACKS, replace=False).tolist())
            df = df[df[group_col].isin(chosen)].copy()
            print(f"[gpu-more] kept {MAX_TRACKS} tracks from {len(keys)} total tracks", flush=True)
        else:
            df = df.copy()
            print(f"[gpu-more] total tracks {len(keys)} <= --max-tracks, keeping all", flush=True)

    # If Track_ID exists, convert it into a clean consecutive Vehicle_ID.
    # This avoids mixing repeated Vehicle_ID across locations/segments.
    if "Track_ID" in df.columns:
        df = df.copy()
        df["Vehicle_ID"] = pd.factorize(df["Track_ID"], sort=True)[0] + 1

    return df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)


def matrix_from_df_track(df: pd.DataFrame, max_frames: int = 620):
    group_col = "Track_ID" if "Track_ID" in df.columns else "Vehicle_ID"
    if group_col == "Track_ID":
        groups_iter = df.groupby("Vehicle_ID", sort=True)
    else:
        groups_iter = df.groupby(group_col, sort=True)

    groups = []
    min_len = None
    for vid, g in groups_iter:
        g = g.sort_values("Frame_ID")
        if len(g) >= r.HIST + r.PRED + 5:
            groups.append((vid, g))
            min_len = len(g) if min_len is None else min(min_len, len(g))

    if not groups:
        raise ValueError("No track has enough continuous frames for selected windows.")

    n = len(groups)
    T = min(int(min_len), max_frames)

    x = np.zeros((n, T), dtype=np.float32)
    y = np.zeros((n, T), dtype=np.float32)
    v = np.zeros((n, T), dtype=np.float32)
    a = np.zeros((n, T), dtype=np.float32)
    lane = np.zeros((n, T), dtype=np.int32)
    vehicle_ids = []

    for i, (vid, g) in enumerate(groups):
        gg = g.iloc[:T]
        vehicle_ids.append(i + 1)
        x[i] = pd.to_numeric(gg["Local_X"], errors="coerce").interpolate().bfill().ffill().to_numpy(np.float32)
        y[i] = pd.to_numeric(gg["Local_Y"], errors="coerce").interpolate().bfill().ffill().to_numpy(np.float32)
        v[i] = pd.to_numeric(gg["v_Vel"], errors="coerce").interpolate().bfill().ffill().to_numpy(np.float32)
        a[i] = pd.to_numeric(gg["v_Acc"], errors="coerce").interpolate().bfill().ffill().to_numpy(np.float32)
        lane[i] = pd.to_numeric(gg["Lane_ID"], errors="coerce").ffill().fillna(1).to_numpy(np.int32) - 1

    print(f"[gpu-more] matrix tracks={n}, frames_per_track={T}", flush=True)
    return np.asarray(vehicle_ids), x, y, v, a, lane


def make_samples_gpu(vehicle_ids, x, y, v, a, lane, stride: int = 12):
    device = dev()
    if device is None or device.type != "cuda":
        print("[gpu-more] CUDA not available for make_samples; falling back to original CPU make_samples", flush=True)
        return _ORIG_MAKE_SAMPLES(vehicle_ids, x, y, v, a, lane, stride=stride)

    n, T = x.shape
    starts = list(range(0, T - r.HIST - r.PRED, stride))
    print(f"[gpu-more] make_samples on GPU: tracks={n}, starts={len(starts)}, expected_samples={n*len(starts)}", flush=True)

    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    vt = torch.as_tensor(v, dtype=torch.float32, device=device)
    lt = torch.as_tensor(lane, dtype=torch.int64, device=device)

    graph_by_start = []

    for si, s in enumerate(starts):
        t = s + r.HIST - 1
        xi = xt[:, t]
        yi = yt[:, t]
        vi = vt[:, t]
        li = lt[:, t]

        graph_mat = np.empty((n, 5), dtype=np.float32)

        for a0 in range(0, n, NEIGHBOR_BATCH):
            a1 = min(n, a0 + NEIGHBOR_BATCH)
            rows = a1 - a0

            xi0 = xi[a0:a1]
            yi0 = yi[a0:a1]
            vi0 = vi[a0:a1]
            li0 = li[a0:a1]

            dx = xi[None, :] - xi0[:, None]
            dy = yi[None, :] - yi0[:, None]
            dist = torch.sqrt(dx * dx + dy * dy)

            mask = dist < 60.0
            local_rows = torch.arange(rows, device=device)
            global_cols = torch.arange(a0, a1, device=device)
            mask[local_rows, global_cols] = False

            count = mask.sum(dim=1).float()
            ahead = mask & (yi[None, :] > yi0[:, None])
            head = yi[None, :] - yi0[:, None]
            big = torch.full_like(head, 1.0e6)
            min_head = torch.where(ahead, head, big).min(dim=1).values
            min_head = torch.where(min_head > 1.0e5, torch.full_like(min_head, 85.0), min_head)
            min_head = torch.clamp(min_head, max=85.0)

            rel_speed_sum = torch.where(mask, vi[None, :] - vi0[:, None], torch.zeros_like(dist)).sum(dim=1)
            mean_rel_speed = rel_speed_sum / torch.clamp(count, min=1.0)

            same_lane = ((li[None, :] == li0[:, None]) & mask).sum(dim=1).float() / 60.0

            graph = torch.stack([
                count / 60.0,
                min_head,
                mean_rel_speed,
                same_lane,
                count
            ], dim=1)

            graph_mat[a0:a1] = graph.detach().cpu().numpy()

            del dx, dy, dist, mask, ahead, head, big, graph

        graph_by_start.append(graph_mat)

        if si % 5 == 0 or si == len(starts) - 1:
            print(f"[gpu-more] graph features done for start {si+1}/{len(starts)}", flush=True)

    samples = []
    for si, s in enumerate(starts):
        graph_mat = graph_by_start[si]
        for i in range(n):
            samples.append({
                "vehicle_id": int(vehicle_ids[i]),
                "start_frame": int(s + 1),
                "hist_xy": np.c_[x[i, s:s+r.HIST], y[i, s:s+r.HIST]],
                "future_xy": np.c_[x[i, s+r.HIST:s+r.HIST+r.PRED], y[i, s+r.HIST:s+r.HIST+r.PRED]],
                "hist_lane": lane[i, s:s+r.HIST],
                "future_lane": lane[i, s+r.HIST:s+r.HIST+r.PRED],
                "hist_acc": a[i, s:s+r.HIST],
                "graph": graph_mat[i].astype(float)
            })

    print(f"[gpu-more] samples made: {len(samples)}", flush=True)
    return samples


def _grad(z):
    out = torch.empty_like(z)
    out[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * r.DT)
    out[:, 0] = (z[:, 1] - z[:, 0]) / r.DT
    out[:, -1] = (z[:, -1] - z[:, -2]) / r.DT
    return out


def _sampen_batch(x, m=2, r_ratio=None):
    B, L = x.shape
    if L <= m + 2:
        return torch.zeros(B, device=x.device)

    if r_ratio is None:
        r_ratio = getattr(r, "SAMPEN_R_RATIO", 0.2)
    sd = x.std(dim=1, unbiased=False)
    tol = torch.clamp(r_ratio * sd, min=1.0e-8)

    def count(mm):
        if L <= mm:
            return torch.zeros(B, device=x.device)
        T = x.unfold(1, mm, 1)
        W = T.shape[1]
        D = (T[:, :, None, :] - T[:, None, :, :]).abs().amax(dim=-1)
        c = (D <= tol[:, None, None]).sum(dim=(1, 2)).float()
        c = (c - W) / 2.0
        return torch.clamp(c, min=0.0)

    cb = count(m)
    ca = count(m + 1)
    return torch.log((cb + 1.0) / (ca + 1.0))


def _mse_batch(speed):
    vals = []
    B, L = speed.shape
    for scale in [1, 2, 3, 4, 5]:
        usable = (L // scale) * scale
        if usable < 8:
            vals.append(torch.zeros(B, device=speed.device))
        else:
            cg = speed[:, :usable].reshape(B, -1, scale).mean(dim=2)
            vals.append(_sampen_batch(cg, m=2))
    return torch.stack(vals, dim=1)


def _dfa_batch(x):
    B, L = x.shape
    y = torch.cumsum(x - x.mean(dim=1, keepdim=True), dim=1)
    sizes = [4, 5, 6, 8, 10, 12, 15]
    Fs = []
    used = []

    for s in sizes:
        nseg = L // s
        if nseg < 2:
            continue
        seg = y[:, :nseg*s].reshape(B, nseg, s)
        t = torch.arange(s, device=x.device, dtype=torch.float32)
        tm = t.mean()
        var_t = torch.sum((t - tm) ** 2).clamp_min(1.0e-8)
        sm = seg.mean(dim=2, keepdim=True)
        slope = ((seg - sm) * (t - tm)).sum(dim=2, keepdim=True) / var_t
        intercept = sm - slope * tm
        trend = slope * t + intercept
        rms = torch.sqrt(torch.mean((seg - trend) ** 2, dim=2) + 1.0e-12)
        F = torch.sqrt(torch.mean(rms ** 2, dim=1) + 1.0e-12)
        Fs.append(F)
        used.append(float(s))

    if len(Fs) < 2:
        return torch.full((B,), 0.5, device=x.device)

    Fmat = torch.stack(Fs, dim=1).clamp_min(1.0e-12)
    logF = torch.log(Fmat)
    logs = torch.log(torch.tensor(used, device=x.device, dtype=torch.float32))
    logs = logs[None, :]
    logs_c = logs - logs.mean(dim=1, keepdim=True)
    logF_c = logF - logF.mean(dim=1, keepdim=True)
    alpha = (logs_c * logF_c).sum(dim=1) / torch.sum(logs_c ** 2)
    return torch.clamp(alpha, 0.0, 2.0)


def _embed_batch(y, m=3, tau=4):
    B, L0 = y.shape
    start = (m - 1) * tau
    cols = []
    for k in range(start, L0):
        cols.append(torch.stack([y[:, k - j * tau] for j in range(m)], dim=1))
    return torch.stack(cols, dim=1)


def _rqa_batch(y, m=3, tau=4):
    E = _embed_batch(y, m=m, tau=tau)
    B, L, M = E.shape
    if L < 3:
        z = torch.zeros(B, device=y.device)
        return z, z, z

    D = torch.cdist(E, E)
    eye = torch.eye(L, dtype=torch.bool, device=y.device)
    D_no_diag = D[:, ~eye].reshape(B, -1)
    eps_ratio = float(getattr(r, "RQA_EPS_PERCENTILE", 15.0)) / 100.0
    k = max(1, int(eps_ratio * D_no_diag.shape[1]))
    eps = torch.kthvalue(D_no_diag, k, dim=1).values

    R = D < eps[:, None, None]
    R[:, eye] = False

    rec = R.sum(dim=(1, 2)).float()
    rr = rec / max(1, L * L - L)

    diag_pairs = (R[:, :-1, :-1] & R[:, 1:, 1:]).sum(dim=(1, 2)).float()
    det = torch.clamp((2.0 * diag_pairs) / torch.clamp(rec, min=1.0), 0.0, 1.0)
    mean_diag = 1.0 + 4.0 * det
    return rr, det, mean_diag


def _lyap_proxy_batch(y, m=3, tau=4):
    E = _embed_batch(y, m=m, tau=tau)
    if E.shape[1] < 4:
        return torch.zeros(y.shape[0], device=y.device)
    d1 = torch.linalg.norm(E[:, 1:] - E[:, :-1], dim=2).clamp_min(1.0e-8)
    d2 = torch.linalg.norm(E[:, 2:] - E[:, :-2], dim=2).clamp_min(1.0e-8)
    val = torch.log(d2 / d1[:, 1:]).mean(dim=1)
    return torch.clamp(val, -4.0, 4.0)


def _feature_chunk(samples, groups, m=3, tau=4, noise=0.0, seed=1):
    device = dev()
    Hnp = np.stack([s["hist_xy"] for s in samples]).astype(np.float32)
    Lnp = np.stack([s["hist_lane"] for s in samples]).astype(np.int64)
    Gnp = np.stack([s["graph"] for s in samples]).astype(np.float32)

    if noise > 0:
        rng = np.random.default_rng(seed)
        Hnp = Hnp + rng.normal(0.0, noise, size=Hnp.shape).astype(np.float32)

    H = torch.as_tensor(Hnp, device=device)
    lanes = torch.as_tensor(Lnp, device=device)
    G = torch.as_tensor(Gnp, device=device)

    xh = H[:, :, 0]
    yh = H[:, :, 1]

    vx = _grad(xh)
    vy = _grad(yh)
    ax = _grad(vx)
    ay = _grad(vy)
    speed = torch.sqrt(vx * vx + vy * vy + 1.0e-12)

    groups = set(groups)
    feats = []
    names = []

    def add(name, val):
        if val.ndim == 1:
            val = val[:, None]
        feats.append(val.float())
        names.append(name)

    if "base" in groups:
        last_x = xh[:, -1]
        last_y = yh[:, -1]
        add("last_x", last_x)
        add("last_y", last_y)

        for j in range(12, 0, -1):
            add(f"hist_dx_t-{j}", xh[:, -j] - last_x)
            add(f"hist_dy_t-{j}", yh[:, -j] - last_y)

        for prefix, arr in [("vx", vx), ("vy", vy), ("ax", ax), ("ay", ay)]:
            add(f"{prefix}_last", arr[:, -1])
            add(f"{prefix}_mean", arr.mean(dim=1))
            add(f"{prefix}_std", arr.std(dim=1, unbiased=False))

        add("lat_range", xh.max(dim=1).values - xh.min(dim=1).values)
        add("lon_displacement", yh[:, -1] - yh[:, 0])
        add("hist_lane_last", lanes[:, -1].float())
        add("hist_lane_change_count", (lanes[:, 1:] != lanes[:, :-1]).sum(dim=1).float())

    if "delay" in groups:
        for k in range(m):
            idx = max(0, r.HIST - 1 - k * tau)
            add(f"delay_y_{k}", yh[:, idx] - yh[:, -1])
            add(f"delay_x_{k}", xh[:, idx] - xh[:, -1])

    if "lyap" in groups:
        add("lyapunov_proxy_y", _lyap_proxy_batch(yh, m=max(2, min(m, 6)), tau=tau))
        ds = speed[:, 1:] - speed[:, :-1]
        complexity = torch.log1p(ds.std(dim=1, unbiased=False)) / (ds.abs().mean(dim=1) + 1.0e-4)
        add("complexity_speed", complexity)
        jerk_y = _grad(ay)
        add("jerk_y_std", jerk_y.std(dim=1, unbiased=False))
        add("acc_abs_p90", torch.quantile(ay.abs(), 0.90, dim=1))

    if "rqa" in groups:
        rr, det, lmean = _rqa_batch(yh, m=max(2, min(m, 6)), tau=tau)
        add("rqa_recurrence_rate", rr)
        add("rqa_determinism", det)
        add("rqa_mean_diag_len", lmean)

    if "entropy" in groups:
        se_y = _sampen_batch(yh, m=2)
        se_v = _sampen_batch(speed, m=2)
        mse = _mse_batch(speed)
        add("sampen_y", se_y)
        add("sampen_speed", se_v)
        for idx in range(5):
            add(f"mse_speed_s{idx+1}", mse[:, idx])
        add("mse_speed_mean", mse.mean(dim=1))
        scale = torch.arange(1, 6, device=device, dtype=torch.float32)
        sc = scale - scale.mean()
        mc = mse - mse.mean(dim=1, keepdim=True)
        slope = (mc * sc).sum(dim=1) / torch.sum(sc ** 2)
        add("mse_speed_slope", slope)
        add("dfa_y_alpha", _dfa_batch(yh))
        add("dfa_speed_alpha", _dfa_batch(speed))

    if "graph" in groups:
        graph_labels = ["density", "min_headway", "mean_rel_speed", "same_lane_density", "neighbor_count"]
        for idx, lab in enumerate(graph_labels):
            add(f"graph_{lab}", G[:, idx])

    if "poly" in groups:
        old_len = len(feats)
        start = max(0, old_len - 20)
        for idx in range(start, old_len):
            add(f"sq_{names[idx]}", feats[idx].squeeze(1) ** 2)

    X = torch.cat(feats, dim=1).detach().cpu().numpy().astype(np.float64)
    return X, names


def feature_matrix_gpu(samples, groups, m=3, tau=4, noise=0.0, seed=1, return_names=False):
    device = dev()
    if device is None or device.type != "cuda":
        print("[gpu-more] CUDA not available for feature_matrix; falling back to original CPU feature_matrix", flush=True)
        return r.feature_matrix(samples, groups, m=m, tau=tau, noise=noise, seed=seed, return_names=return_names)

    rows = []
    names_ref = None
    total = len(samples)
    for a0 in range(0, total, GPU_BATCH):
        a1 = min(total, a0 + GPU_BATCH)
        Xc, names = _feature_chunk(samples[a0:a1], groups, m=m, tau=tau, noise=noise, seed=seed + a0)
        rows.append(Xc)
        if names_ref is None:
            names_ref = names
        print(f"[gpu-more] feature_matrix GPU {a1}/{total}", flush=True)

    X = np.vstack(rows)
    if return_names:
        return X, names_ref
    return X


class TorchRidge:
    def __init__(self, lam: float = 10.0):
        self.lam = float(lam)

    def fit(self, X, Y):
        device = dev()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
        self.mu = X_t.mean(dim=0)
        self.sd = X_t.std(dim=0, unbiased=False) + 1.0e-6
        Z = (X_t - self.mu) / self.sd
        Z = torch.cat([torch.ones((Z.shape[0], 1), device=device), Z], dim=1)
        I = torch.eye(Z.shape[1], device=device)
        I[0, 0] = 0.0
        self.W = torch.linalg.solve(Z.T @ Z + self.lam * I, Z.T @ Y_t)
        return self

    def predict(self, X):
        device = self.W.device
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        Z = (X_t - self.mu) / self.sd
        Z = torch.cat([torch.ones((Z.shape[0], 1), device=device), Z], dim=1)
        return (Z @ self.W).detach().cpu().numpy().astype(float)


class TorchLogistic:
    def fit(self, X, y, lr: float = 0.06, epochs: int = 900, lam: float = 0.004):
        device = dev()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.mu = X_t.mean(dim=0)
        self.sd = X_t.std(dim=0, unbiased=False) + 1.0e-6
        Z = (X_t - self.mu) / self.sd
        Z = torch.cat([torch.ones((Z.shape[0], 1), device=device), Z], dim=1)
        w = torch.zeros(Z.shape[1], dtype=torch.float32, device=device)

        pos = max(1.0, float(y_t.sum().item()))
        neg = max(1.0, float(len(y) - y_t.sum().item()))
        weights = torch.where(y_t == 1, torch.tensor(neg / pos, device=device), torch.tensor(1.0, device=device))
        weights = weights / weights.mean()

        for _ in range(epochs):
            p = torch.sigmoid(torch.clamp(Z @ w, -30.0, 30.0))
            reg = torch.cat([torch.zeros(1, device=device), w[1:]])
            grad = Z.T @ ((p - y_t) * weights) / len(y_t) + lam * reg
            w = w - lr * grad

        self.w = w
        return self

    def proba(self, X):
        device = self.w.device
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        Z = (X_t - self.mu) / self.sd
        Z = torch.cat([torch.ones((Z.shape[0], 1), device=device), Z], dim=1)
        p = torch.sigmoid(torch.clamp(Z @ self.w, -30.0, 30.0))
        return p.detach().cpu().numpy().astype(float)


def cv_prediction_gpu(samples, acceleration=False, noise=0.0, seed=1):
    device = dev()
    H = np.stack([s["hist_xy"] for s in samples]).astype(np.float32)
    if noise > 0:
        rng = np.random.default_rng(seed)
        H = H + rng.normal(0.0, noise, H.shape).astype(np.float32)

    h = torch.as_tensor(H, dtype=torch.float32, device=device)
    v1 = (h[:, -1, :] - h[:, -6, :]) / (5.0 * r.DT)
    k = torch.arange(1, r.PRED + 1, device=device, dtype=torch.float32)[:, None]
    if acceleration:
        v0 = (h[:, -6, :] - h[:, -11, :]) / (5.0 * r.DT)
        acc = (v1 - v0) / (5.0 * r.DT)
        pred = h[:, None, -1, :] + v1[:, None, :] * k[None, :, :] * r.DT + 0.5 * acc[:, None, :] * (k[None, :, :] * r.DT) ** 2
    else:
        pred = h[:, None, -1, :] + v1[:, None, :] * k[None, :, :] * r.DT
    return pred.reshape(len(samples), 2 * r.PRED).detach().cpu().numpy().astype(float)


def residual_to_basis_gpu(R):
    device = dev()
    R_t = torch.as_tensor(R, dtype=torch.float32, device=device).reshape(-1, r.PRED, 2)
    u = torch.linspace(1.0 / r.PRED, 1.0, r.PRED, device=device)
    B = torch.stack([u, u * u, u * u * u], dim=1)
    pinv = torch.linalg.pinv(B)
    cx = R_t[:, :, 0] @ pinv.T
    cy = R_t[:, :, 1] @ pinv.T
    return torch.cat([cx, cy], dim=1).detach().cpu().numpy().astype(float)


def basis_to_residual_gpu(C):
    device = dev()
    C_t = torch.as_tensor(C, dtype=torch.float32, device=device)
    u = torch.linspace(1.0 / r.PRED, 1.0, r.PRED, device=device)
    B = torch.stack([u, u * u, u * u * u], dim=1)
    k = B.shape[1]
    rx = C_t[:, :k] @ B.T
    ry = C_t[:, k:2*k] @ B.T
    out = torch.stack([rx, ry], dim=2).reshape(C_t.shape[0], 2 * r.PRED)
    return out.detach().cpu().numpy().astype(float)


def metric_dict_gpu(Y, P):
    device = dev()
    yt = torch.as_tensor(Y, dtype=torch.float32, device=device).reshape(-1, r.PRED, 2)
    yp = torch.as_tensor(P, dtype=torch.float32, device=device).reshape(-1, r.PRED, 2)
    d = torch.sqrt(torch.sum((yt - yp) ** 2, dim=2))
    return {
        "ADE": float(d.mean().detach().cpu()),
        "FDE": float(d[:, -1].mean().detach().cpu()),
        "RMSE_x": float(torch.sqrt(torch.mean((yt[:, :, 0] - yp[:, :, 0]) ** 2)).detach().cpu()),
        "RMSE_y": float(torch.sqrt(torch.mean((yt[:, :, 1] - yp[:, :, 1]) ** 2)).detach().cpu()),
        "RMSE": float(torch.sqrt(torch.mean((yt - yp) ** 2)).detach().cpu()),
        "Lat_MAE": float(torch.mean(torch.abs(yt[:, :, 0] - yp[:, :, 0])).detach().cpu()),
        "Lon_MAE": float(torch.mean(torch.abs(yt[:, :, 1] - yp[:, :, 1])).detach().cpu()),
    }


def main():
    global MAX_TRACKS, SEED, GPU_BATCH, NEIGHBOR_BATCH, DEVICE_NAME

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default=None)
    ap.add_argument("--input-units", default="feet", choices=["feet", "meters"])
    ap.add_argument("--output", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--xgb-estimators", type=int, default=80)
    ap.add_argument("--n-vehicles", type=int, default=24)
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--n-lanes", type=int, default=4)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--skip-figures", action="store_true")

    ap.add_argument("--max-tracks", type=int, default=3000)
    ap.add_argument("--gpu-preprocess", action="store_true")
    ap.add_argument("--gpu-features", action="store_true")
    ap.add_argument("--gpu-batch", type=int, default=4096)
    ap.add_argument("--neighbor-batch", type=int, default=2048)

    args = ap.parse_args()

    MAX_TRACKS = int(args.max_tracks)
    SEED = int(args.seed)
    GPU_BATCH = int(args.gpu_batch)
    NEIGHBOR_BATCH = int(args.neighbor_batch)
    DEVICE_NAME = "cuda" if args.device in ("cuda", "auto") else "cpu"

    if torch is None:
        raise RuntimeError("torch is required for run_experiments_gpu_more.py")

    print("[gpu-more] torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("[gpu-more] GPU:", torch.cuda.get_device_name(0), flush=True)

    r.load_real_ngsim_csv = load_real_ngsim_csv_limited
    r.matrix_from_df = matrix_from_df_track

    if args.gpu_preprocess:
        r.make_samples = make_samples_gpu

    if args.gpu_features:
        r.feature_matrix = feature_matrix_gpu

    r.StandardRidge = TorchRidge
    r.WeightedLogistic = TorchLogistic
    r.cv_prediction = cv_prediction_gpu
    r.residual_to_basis = residual_to_basis_gpu
    r.basis_to_residual = basis_to_residual_gpu
    r.metric_dict = metric_dict_gpu

    r.run(args)


if __name__ == "__main__":
    main()
