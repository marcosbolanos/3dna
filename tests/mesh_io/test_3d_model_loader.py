from pathlib import Path

import numpy as np
import pytest
import trimesh

from threedna.mesh_io import model_loader


def test_loader_skips_reconstruction_for_watertight_mesh(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    watertight_mesh = trimesh.creation.box()

    def fake_load(_: Path) -> trimesh.Trimesh:
        return watertight_mesh

    monkeypatch.setattr(model_loader.trimesh, "load", fake_load)

    loaded = model_loader.load_3d_model(Path("dummy.glb"))
    stderr = capsys.readouterr().err

    assert isinstance(loaded, trimesh.Trimesh)
    assert loaded.is_watertight
    assert stderr == ""


def test_loader_warns_for_non_watertight_mesh(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_mesh = trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2]],
        process=False,
    )

    def fake_load(_: Path) -> trimesh.Trimesh:
        return source_mesh

    monkeypatch.setattr(model_loader.trimesh, "load", fake_load)

    loaded = model_loader.load_3d_model(Path("dummy.glb"))
    stderr = capsys.readouterr().err

    assert loaded is not source_mesh
    assert not loaded.is_watertight
    assert "\033[31m" in stderr
    assert "not watertight" in stderr
    assert "included watertight helper tool" in stderr


def test_loader_scales_mesh_using_axis_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_mesh = trimesh.creation.box(extents=[2.0, 4.0, 6.0])

    def fake_load(_: Path) -> trimesh.Trimesh:
        return source_mesh

    monkeypatch.setattr(model_loader.trimesh, "load", fake_load)

    loaded = model_loader.load_3d_model(
        Path("dummy.glb"),
        scale_axis="x",
        target_length_nm=20.0,
    )

    extents = np.asarray(loaded.bounding_box.extents)
    assert np.isclose(extents[0], 20.0)
    assert np.allclose(extents, np.array([20.0, 40.0, 60.0]))


def test_loader_rejects_partial_scaling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watertight_mesh = trimesh.creation.box()

    def fake_load(_: Path) -> trimesh.Trimesh:
        return watertight_mesh

    monkeypatch.setattr(model_loader.trimesh, "load", fake_load)

    with pytest.raises(ValueError, match="provided together"):
        model_loader.load_3d_model(Path("dummy.glb"), scale_axis="z")
