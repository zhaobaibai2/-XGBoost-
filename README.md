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

---

## Data Download Instructions

This project uses NGSIM vehicle trajectory data.  
Because the original data files and generated intermediate files can be very large, they are not directly uploaded to this GitHub repository.

The GitHub repository mainly contains:

- source code
- README documentation
- the data download script
- configuration files
- lightweight project files

Large data files should be downloaded locally by the user.

## How to Download the Data

A helper script is provided in this repository:

    download_ngsim_sample.py

To download the dataset used in this project, run:

    python download_ngsim_sample.py --out data/raw/ngsim_sample.csv --limit 10000000

The argument meaning is:

- `--out data/raw/ngsim_sample.csv`: save the downloaded data to `data/raw/ngsim_sample.csv`
- `--limit 10000000`: download up to 10,000,000 rows
- the downloaded CSV file may be large, so it is intentionally not committed to GitHub

If you only want to quickly test whether the code works, you can use a smaller limit:

    python download_ngsim_sample.py --out data/raw/ngsim_sample.csv --limit 20000

## Important Notes

Before running the project, please download the data first:

    python download_ngsim_sample.py --out data/raw/ngsim_sample.csv --limit 10000000

After the download finishes, the data file should be located at:

    data/raw/ngsim_sample.csv

The `data/` directory is ignored by Git through `.gitignore`, so downloaded data will stay on the local machine and will not be uploaded to GitHub.

This design keeps the repository lightweight while still allowing others to reproduce the experiment by downloading the data themselves.

