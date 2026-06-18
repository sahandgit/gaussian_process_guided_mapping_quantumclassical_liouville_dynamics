from pathlib import Path
import json


def test_console_scripts_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "gp-mqcld"' in pyproject
    assert 'gp-mqcld = "gp_mint_qcle.cli:main"' in pyproject
    assert 'gp-mqcld-run = "gp_mint_qcle.run:main"' in pyproject
    assert 'gp-mqcld-compare = "gp_mint_qcle.Compare_gp_se_qcle:main"' in pyproject
    assert 'gp-mqcld-smoke = "gp_mint_qcle.cli_smoke:main"' in pyproject
    # Backward-compatible aliases remain available for older notes/scripts.
    assert 'gp-mint = "gp_mint_qcle.cli:main"' in pyproject
    assert 'liouvillegp = "gp_mint_qcle.cli:main"' in pyproject


def test_public_import_alias():
    import gp_mqcld as gpm
    from gp_mqcld.Models import TullyModel
    assert hasattr(gpm, "__version__")
    assert TullyModel is not None


def test_legacy_import_alias_still_available():
    import liouvillegp_mint as lgm
    from liouvillegp_mint.Models import TullyModel
    assert hasattr(lgm, "__version__")
    assert TullyModel is not None


def test_notebooks_exist_and_are_valid_json():
    nb_dir = Path("notebooks")
    names = [
        "00_installation_and_smoke_test.ipynb",
        "01_pipeline_walkthrough.ipynb",
        "02_run_full_density_p0_cases.ipynb",
        "03_load_results_and_basic_diagnostics.ipynb",
        "04_compare_se_qcle_gp.ipynb",
    ]
    for name in names:
        path = nb_dir / name
        assert path.exists(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
