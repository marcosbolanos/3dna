from pathlib import Path

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
