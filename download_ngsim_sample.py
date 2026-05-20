# -*- coding: utf-8 -*-
"""
Download a small NGSIM sample from the official USDOT Socrata endpoint.

Example:
    python download_ngsim_sample.py --out data/raw/ngsim_sample.csv --limit 20000

The full dataset is large. This helper intentionally downloads only selected fields
and a configurable number of rows so the course report can be reproduced quickly.
"""
from __future__ import annotations
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://data.transportation.gov/resource/8ect-6jqj.csv"
FIELDS = [
    "vehicle_id", "frame_id", "total_frames", "global_time",
    "local_x", "local_y", "v_vel", "v_acc", "lane_id",
    "preceding", "following", "space_headway", "location"
]


def build_url(limit: int, where: str | None = None):
    params = {
        "$select": ",".join(FIELDS),
        "$limit": str(limit),
        "$order": "vehicle_id,frame_id"
    }
    if where:
        params["$where"] = where
    return API + "?" + urllib.parse.urlencode(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/ngsim_sample.csv")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--where", default=None, help="Optional Socrata where clause, e.g. \"vehicle_id <= 50\".")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = build_url(args.limit, args.where)
    print("Downloading:", url)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    out.write_bytes(data)
    print(f"Saved {len(data)} bytes to {out}")


if __name__ == "__main__":
    main()
