from dataclasses import dataclass

import numpy as np
import trimesh

from threedna.surface_projection import project_points_to_mesh
from threedna._geodesic_cpp import GeodesicMesh as _GeodesicMesh


@dataclass(frozen=True)
class GeodesicCurvePath:
    points: np.ndarray
    vertex_indices: np.ndarray


def _as_curve_points(curve: np.ndarray) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("curve must be a 2D array")
    if arr.shape[0] == 3:
        points = arr.T
    elif arr.shape[1] == 3:
        points = arr
    else:
        raise ValueError("curve must have shape (3, n_points) or (n_points, 3)")
    if len(points) < 3:
        raise ValueError("curve must contain at least 3 points")
    return points


def _trace_exact_geodesic_path(
    geodesic_mesh: _GeodesicMesh,
    source_idx: int,
    target_idx: int,
    vertex_positions: np.ndarray,
) -> np.ndarray:
    """Trace exact geodesic path between two vertices using C++ MMP algorithm."""
    path = geodesic_mesh.trace_path(source_idx, target_idx)

    points = []
    for typ, idx, bary in path:
        if typ == 0:
            points.append(vertex_positions[idx])
        elif typ == 1:
            edge_vertices = geodesic_mesh.edge_vertex_indices()[idx]
            p0 = vertex_positions[edge_vertices[0]]
            p1 = vertex_positions[edge_vertices[1]]
            point = (1 - bary[0]) * p0 + bary[0] * p1
            points.append(point)
        elif typ == 2:
            face_vertices = geodesic_mesh.face_vertex_indices()[idx]
            p0 = vertex_positions[face_vertices[0]]
            p1 = vertex_positions[face_vertices[1]]
            p2 = vertex_positions[face_vertices[2]]
            point = bary[0] * p0 + bary[1] * p1 + bary[2] * p2
            points.append(point)

    return np.array(points, dtype=np.float64)


def reconstruct_geodesic_curve_on_mesh(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
) -> GeodesicCurvePath:
    """
    Reconstruct a closed curve as exact geodesic paths between consecutive nodes.
    Uses the MMP exact geodesic algorithm from geometry-central via C++.
    """
    nodes = _as_curve_points(curve)
    projected_nodes = project_points_to_mesh(mesh, nodes)

    _, nearest_vertex = mesh.kdtree.query(projected_nodes)
    node_vertex_idx = np.asarray(nearest_vertex, dtype=np.int64)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    geodesic_mesh = _GeodesicMesh(V, F)
    vertex_positions = geodesic_mesh.vertex_positions()

    segment_paths = []
    n_nodes = len(node_vertex_idx)
    for i in range(n_nodes):
        source = int(node_vertex_idx[i])
        target = int(node_vertex_idx[(i + 1) % n_nodes])

        path_points = _trace_exact_geodesic_path(
            geodesic_mesh, source, target, vertex_positions
        )
        segment_paths.append(path_points)

    stitched_points = np.vstack(segment_paths)

    vertex_indices = []
    for typ, idx, _ in segment_paths[0]:
        vertex_indices.append(idx)
    for path in segment_paths[1:]:
        for typ, idx, _ in path[1:]:
            vertex_indices.append(idx)
    vertex_indices = np.array(vertex_indices, dtype=np.int64)

    return GeodesicCurvePath(points=stitched_points, vertex_indices=vertex_indices)
