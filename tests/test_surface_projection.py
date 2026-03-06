import numpy as np
import trimesh

from threedna.surface_projection import project_points_to_mesh


def test_project_points_to_mesh_projects_to_triangle_plane() -> None:
    mesh = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    points = np.array(
        [
            [0.5, 0.5, 1.0],
            [0.25, 0.75, -1.0],
        ],
        dtype=np.float64,
    )

    projected = project_points_to_mesh(mesh, points)

    assert projected.shape == points.shape
    assert np.allclose(projected[:, 2], 0.0)
    assert np.allclose(projected[:, :2], points[:, :2])


def test_project_points_to_mesh_clamps_outside_triangle() -> None:
    mesh = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    points = np.array(
        [
            [2.0, 2.0, 3.0],
            [1.5, 1.5, -2.0],
        ],
        dtype=np.float64,
    )

    projected = project_points_to_mesh(mesh, points)

    expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
    assert np.allclose(projected, expected)
