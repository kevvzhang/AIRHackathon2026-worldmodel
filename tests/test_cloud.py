"""The Modal app must at least import: it is only exercised for real in the cloud."""

import pytest

pytest.importorskip("modal")


def test_cloud_app_is_well_formed():
    import cloud

    assert cloud.app.name == "sandwm"
    for name in ("generate", "train", "evaluate", "sweep", "reevaluate"):
        assert hasattr(cloud, name), f"cloud.py lost its {name!r} entry point"
    assert isinstance(cloud.GPU, list) and cloud.GPU
