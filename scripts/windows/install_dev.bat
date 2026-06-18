@echo off
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
pytest
gp-mqcld-smoke
