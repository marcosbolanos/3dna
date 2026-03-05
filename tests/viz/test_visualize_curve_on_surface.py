from pathlib import Path

import numpy as np
import trimesh

from threedna.viz.visualize_curve_on_surface import (
    build_curve_on_surface_scene,
    render_curve_on_surface,
)


def test_build_curve_on_surface_scene_accepts_3xn_curve() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    curve = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(
        np.float32
    )

    scene = build_curve_on_surface_scene(mesh=mesh, curve=curve)

    assert len(scene.geometry) == 2
    assert "surface_mesh" in scene.geometry
    assert "initialized_ring" in scene.geometry


def test_render_curve_on_surface_exports_glb(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    curve = np.array(
        [
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    output = tmp_path / "curve_on_surface.glb"
    render_curve_on_surface(mesh=mesh, curve=curve, output_glb=output)

    assert output.exists()
