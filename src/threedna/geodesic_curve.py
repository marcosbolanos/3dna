from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
import trimesh

from threedna.surface_projection import project_points_to_mesh


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


def _mesh_edge_graph(mesh: trimesh.Trimesh) -> csr_matrix:
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    n_vertices = len(mesh.vertices)
    if len(edges) == 0:
        raise ValueError("mesh has no edges")

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([lengths, lengths])
    return csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))


def _recover_vertex_path(
    predecessors: np.ndarray,
    source_idx: int,
    target_idx: int,
) -> np.ndarray:
    if source_idx == target_idx:
        return np.array([source_idx], dtype=np.int64)

    current = target_idx
    reverse_path = [current]
    max_steps = len(predecessors) + 1
    for _ in range(max_steps):
        current = int(predecessors[current])
        if current < 0:
            return np.array([source_idx, target_idx], dtype=np.int64)
        reverse_path.append(current)
        if current == source_idx:
            break
    reverse_path.reverse()
    return np.asarray(reverse_path, dtype=np.int64)


def reconstruct_geodesic_curve_on_mesh(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
) -> GeodesicCurvePath:
    """
    Reconstruct a closed curve as mesh-edge geodesic paths between consecutive nodes.

    This is an edge-graph geodesic approximation: each segment is a shortest path
    on the mesh vertex graph.
    """
    nodes = _as_curve_points(curve)
    projected_nodes = project_points_to_mesh(mesh, nodes)

    _, nearest_vertex = mesh.kdtree.query(projected_nodes)
    node_vertex_idx = np.asarray(nearest_vertex, dtype=np.int64)

    graph = _mesh_edge_graph(mesh)
    n_nodes = len(node_vertex_idx)
    segment_vertex_paths: list[np.ndarray] = []

    for i in range(n_nodes):
        source = int(node_vertex_idx[i])
        target = int(node_vertex_idx[(i + 1) % n_nodes])
        _, predecessors = csgraph.dijkstra(
            graph,
            directed=False,
            indices=source,
            return_predecessors=True,
        )
        segment_vertex_paths.append(_recover_vertex_path(predecessors, source, target))

    stitched_indices = segment_vertex_paths[0].copy()
    for path in segment_vertex_paths[1:]:
        if len(path) == 0:
            continue
        stitched_indices = np.concatenate([stitched_indices, path[1:]])

    if stitched_indices[0] != stitched_indices[-1]:
        stitched_indices = np.concatenate([stitched_indices, stitched_indices[:1]])

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    stitched_points = vertices[stitched_indices]
    return GeodesicCurvePath(points=stitched_points, vertex_indices=stitched_indices)
