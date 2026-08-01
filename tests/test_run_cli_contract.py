from pathlib import Path


def test_run_requires_explicit_density_mode():
    """The architecture choice must be visible in every production command."""
    source = Path(__file__).resolve().parent / "run.py"
    if not source.exists():
        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "gp_mint_qcle" / "run.py"
        )
    text = source.read_text(encoding="utf-8")
    argument = text[
        text.find('p.add_argument("--density_mode"'):
        text.find('p.add_argument("--sampling_mode"')
    ]
    assert 'choices=["full", "diff"], required=True' in argument
    assert "No default" in argument
