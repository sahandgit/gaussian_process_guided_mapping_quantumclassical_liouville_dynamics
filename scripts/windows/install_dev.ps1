# Run from the repository root.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
pytest
gp-mqcld-smoke
