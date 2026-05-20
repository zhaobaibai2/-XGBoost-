# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

FT_TO_M = 0.3048

def standardize_columns(df):
    lower = {str(c).lower(): c for c in df.columns}
    mapping = {
        "vehicle_id": "Vehicle_ID",
        "frame_id": "Frame_ID",
        "total_frames": "Total_Frames",
        "global_time": "Global_Time",
        "local_x": "Local_X",
        "local_y": "Local_Y",
        "v_vel": "v_Vel",
        "v_acc": "v_Acc",
        "lane_id": "Lane_ID",
        "preceding": "Preceding",
        "following": "Following",
        "space_headway": "Space_Headway",
        "location": "Location",
    }
    rename = {}
    for low, canon in mapping.items():
        if canon not in df.columns and low in lower:
            rename[lower[low]] = canon
    return df.rename(columns=rename)

def clean(input_csv, out_csv, input_units="feet", min_len=400, max_jump_m=8.0):
    df = pd.read_csv(input_csv)
    df = standardize_columns(df)

    required = ["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    for c in ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y"]).copy()

    if "Location" not in df.columns:
        df["Location"] = "unknown"
    if "Global_Time" not in df.columns:
        df["Global_Time"] = df["Frame_ID"] * 100
    if "Lane_ID" not in df.columns:
        df["Lane_ID"] = np.clip(np.round(df["Local_X"] / 12.0) + 1, 1, 8)

    if "v_Vel" not in df.columns:
        df["v_Vel"] = np.nan
    if "v_Acc" not in df.columns:
        df["v_Acc"] = np.nan

    if input_units == "feet":
        for c in ["Local_X", "Local_Y", "v_Vel", "v_Acc", "Space_Headway"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * FT_TO_M
    elif input_units != "meters":
        raise ValueError("--input-units must be feet or meters")

    tracks = []
    new_id = 1

    group_cols = ["Location", "Vehicle_ID"]
    for _, g in df.sort_values(group_cols + ["Global_Time", "Frame_ID"]).groupby(group_cols, sort=False):
        g = g.sort_values(["Global_Time", "Frame_ID"]).copy()

        frame_gap = g["Frame_ID"].diff()
        time_gap = g["Global_Time"].diff()
        dx = g["Local_X"].diff()
        dy = g["Local_Y"].diff()
        step = np.sqrt(dx * dx + dy * dy)

        # 新轨迹段条件：帧不连续、时间不连续、或者相邻两帧位移过大。
        new_seg = (
            frame_gap.isna()
            | (frame_gap <= 0)
            | (frame_gap > 2)
            | (time_gap <= 0)
            | (time_gap > 250)
            | (step > max_jump_m)
        )
        seg_id = new_seg.cumsum()

        for _, s in g.groupby(seg_id):
            if len(s) < min_len:
                continue

            s = s.copy()
            s["Original_Vehicle_ID"] = s["Vehicle_ID"].astype(int)
            s["Original_Location"] = s["Location"].astype(str)
            s["Vehicle_ID"] = new_id
            s["Frame_ID"] = np.arange(1, len(s) + 1)
            s["Total_Frames"] = len(s)
            s["Global_Time"] = np.arange(len(s)) * 100

            # 速度/加速度缺失时用 Local_Y 重新估计
            s["v_Vel"] = pd.to_numeric(s["v_Vel"], errors="coerce")
            if s["v_Vel"].isna().all():
                s["v_Vel"] = s["Local_Y"].diff().fillna(0) / 0.1
            else:
                s["v_Vel"] = s["v_Vel"].interpolate().bfill().ffill()

            s["v_Acc"] = pd.to_numeric(s["v_Acc"], errors="coerce")
            if s["v_Acc"].isna().all():
                s["v_Acc"] = s["v_Vel"].diff().fillna(0) / 0.1
            else:
                s["v_Acc"] = s["v_Acc"].interpolate().bfill().ffill()

            for c in ["Preceding", "Following"]:
                if c not in s.columns:
                    s[c] = 0
            if "Space_Headway" not in s.columns:
                s["Space_Headway"] = np.nan

            tracks.append(s)
            new_id += 1

    if not tracks:
        raise RuntimeError(
            "No continuous tracks kept. Try lowering --min-len to 200 or downloading more rows."
        )

    out = pd.concat(tracks, ignore_index=True)
    cols = [
        "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
        "Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID",
        "Preceding", "Following", "Space_Headway", "Location",
        "Original_Vehicle_ID", "Original_Location"
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(["Vehicle_ID", "Frame_ID"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"saved: {out_csv}")
    print(f"rows: {len(out)}")
    print(f"tracks: {out.Vehicle_ID.nunique()}")
    print(out.groupby("Vehicle_ID").size().describe())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/raw/ngsim_clean_tracks.csv")
    ap.add_argument("--input-units", default="feet", choices=["feet", "meters"])
    ap.add_argument("--min-len", type=int, default=400)
    ap.add_argument("--max-jump-m", type=float, default=8.0)
    args = ap.parse_args()
    clean(args.input, args.out, args.input_units, args.min_len, args.max_jump_m)
