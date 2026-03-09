import numpy as np
import trimesh

from threedna._surface_kernels_cpp import (
    project_points_to_mesh as _project_points_to_mesh_cpp,
)


def project_points_to_mesh(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Project points onto the mesh surface using C++ implementation."""
    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("mesh contains no triangles")

    projected = _project_points_to_mesh_cpp(vertices, faces, query)
    return np.asarray(projected, dtype=np.float64)
