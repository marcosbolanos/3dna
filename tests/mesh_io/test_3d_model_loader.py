from pathlib import Path

import pytest
import trimesh

from threedna.mesh_io import model_loader


def test_loader_skips_reconstruction_for_watertight_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watertight_mesh = trimesh.creation.box()

    def fake_load(_: Path) -> trimesh.Trimesh:
        return watertight_mesh

    def should_not_be_called(**_: object) -> trimesh.Trimesh:
        raise AssertionError("_make_watertight_open3d should not be called")

    monkeypatch.setattr(model_loader.trimesh, "load", fake_load)
    monkeypatch.setattr(model_loader, "_make_watertight_open3d", should_not_be_called)

    loaded = model_loader.load_3d_model(Path("dummy.glb"))

    assert isinstance(loaded, trimesh.Trimesh)
    assert loaded.is_watertight


def test_loader_calls_reconstruction_for_non_watertight_mesh(
    monkeypatch: pytest.MonkeyPatch,
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
    reconstructed = trimesh.creation.box()
    captured: dict[str, object] = {}

    def fake_load(_: Path) -> trimesh.Trimesh:
        return source_mesh

    def fake_reconstruct(**kwargs: object) -> trimesh.Trimesh:
        captured.update(kwargs)
        return reconstructed

    monkeypatch.setattr(model_loader.trimesh, "load", fake_load)
    monkeypatch.setattr(model_loader, "_make_watertight_open3d", fake_reconstruct)

    loaded = model_loader.load_3d_model(Path("dummy.glb"))

    assert loaded is reconstructed
    assert captured["mesh"] is not source_mesh
    assert captured["sample_points"] == 250000
    assert captured["poisson_depth"] == 9
    assert captured["density_trim_quantile"] == 0.0
    assert captured["target_faces"] == 15000
