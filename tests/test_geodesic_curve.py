import numpy as np
import trimesh

from threedna.geodesic_curve import reconstruct_geodesic_curve_on_mesh
from threedna.surface_projection import project_points_to_mesh


def test_reconstruct_geodesic_curve_returns_closed_path() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    curve = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(
        np.float32
    )

    geodesic = reconstruct_geodesic_curve_on_mesh(mesh, curve)

    assert geodesic.points.shape[1] == 3
    assert geodesic.vertex_indices.ndim == 1
    assert len(geodesic.points) == len(geodesic.vertex_indices)
    assert geodesic.vertex_indices[0] == geodesic.vertex_indices[-1]
    assert np.allclose(geodesic.points[0], geodesic.points[-1])


def test_reconstructed_curve_points_stay_on_mesh_surface() -> None:
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    curve = np.array(
        [
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    geodesic = reconstruct_geodesic_curve_on_mesh(mesh, curve)
    projected = project_points_to_mesh(mesh, geodesic.points)
    dist = np.linalg.norm(geodesic.points - projected, axis=1)

    assert float(dist.max()) < 1e-7
