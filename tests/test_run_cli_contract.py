from pathlib import Path


def test_run_default_density_mode_is_full():
    src = Path(__file__).resolve().parents[1] / "src" / "gp_mint_qcle" / "run.py"
    text = src.read_text(encoding="utf-8")
    assert 'choices=["full", "diff"], default="full"' in text
    assert "REQUIRED" not in text[text.find('p.add_argument("--density_mode"'):text.find('p.add_argument("--sampling_mode"')]
