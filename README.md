# Autodrive Nonlinear Trajectory Report Code

This folder contains the scripts and saved outputs used by the LaTeX report.

## Environment

```bash
conda activate real_car
pip install -r requirements.txt
```

GPU is preferred. The main run metadata under `runs/ngsim_8000_tracks/results/`
records `xgboost_regressor_used_device: cuda`.

## Data

The source dataset is USDOT ITS DataHub NGSIM Vehicle Trajectories and
Supporting Data, DOI `10.21949/1504477`. Put raw CSV files under
`data/raw/`, then clean them with:

```bash
python clean_ngsim_tracks.py --input data/raw/ngsim_raw.csv \
  --out data/raw/ngsim_clean_tracks.csv --input-units feet --min-len 400
```

## Main Run

To rerun the full 8000-track experiment with GPU acceleration:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python run_experiments_gpu_more.py --input-csv data/raw/ngsim_clean_tracks.csv \
  --input-units meters --output runs/ngsim_8000_tracks --device cuda \
  --max-tracks 8000 --stride 16 --seed 42 --xgb-estimators 80 \
  --gpu-preprocess --gpu-features --skip-figures
```

The saved report results are under `runs/ngsim_8000_tracks/`. To regenerate
figures and evidence tables from the saved outputs:

```bash
python make_publication_figures.py --run-dir runs/ngsim_8000_tracks \
  --report-fig-dir ../figures --table-dir ../tables
python make_extra_analysis.py --run-dir runs/ngsim_8000_tracks \
  --report-fig-dir ../figures --table-dir ../tables
```

The top-level `scripts/run_all.sh` wraps GPU checking, figure/table generation,
and LaTeX compilation.
# -XGBoost-
