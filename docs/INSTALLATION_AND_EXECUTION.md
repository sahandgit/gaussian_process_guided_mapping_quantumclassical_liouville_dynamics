# Installation and Execution

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
pytest
```

## Smoke test

```powershell
gp-mqcld-smoke
```

## Run

```powershell
gp-mqcld-run --density_mode full --P0 10 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --n_train 2000 --out ".\runs\P0_10"
```

## Compare

```powershell
gp-mqcld-compare ".\runs" --p0-list 10 100 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --density-times "0,500,800,1500" --out ".\runs\comparison_P0_10_100"
```
