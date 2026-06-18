def test_public_imports():
    import gp_mint_qcle as gmq
    assert gmq.D == 6
    assert gmq.__version__


def test_core_module_imports():
    modules = [
        "Models", "Mint", "Sampling", "GP_Density", "GPDerivatives",
        "Monodromy", "Operator", "Observables", "Collector", "Dynamics",
        "Visualization", "qcle_grid_tully", "Compare_gp_se_qcle",
    ]
    for name in modules:
        __import__(f"gp_mint_qcle.{name}")
