from dataclasses import dataclass

import numpy as np
import trimesh

from threedna._bindings._geodesic_cpp import GeodesicMesh, SurfacePoint
from threedna._bindings._surface_kernels_cpp import (
    project_points_to_surface_points as _project_points_to_surface_points,
)


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


def _surface_point_to_xyz(
    sp: SurfacePoint,
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    if sp.type == 0:
        return vertices[sp.index]
    if sp.type == 1:
        edge = edges[sp.index]
        t = float(sp.coords[0])
        return (1.0 - t) * vertices[edge[0]] + t * vertices[edge[1]]
    if sp.type == 2:
        face = faces[sp.index]
        b0, b1, b2 = float(sp.coords[0]), float(sp.coords[1]), float(sp.coords[2])
        return b0 * vertices[face[0]] + b1 * vertices[face[1]] + b2 * vertices[face[2]]
    raise ValueError("unknown surface point type")


def _project_to_face_surface_points(
    mesh: trimesh.Trimesh, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_ids, bary, _ = _project_points_to_surface_points(vertices, faces, nodes)
    return np.asarray(face_ids, dtype=np.int64), np.asarray(bary, dtype=np.float64)


def reconstruct_geodesic_curve_on_mesh(
    mesh: trimesh.Trimesh, curve: np.ndarray
) -> GeodesicCurvePath:
    nodes = _as_curve_points(curve)

    tri_ids, bary_coords = _project_to_face_surface_points(mesh, nodes)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    geodesic_mesh = GeodesicMesh(vertices, faces)
    gc_vertices = geodesic_mesh.vertex_positions()
    gc_faces = geodesic_mesh.face_vertex_indices()
    gc_edges = geodesic_mesh.edge_vertex_indices()

    path_points: list[np.ndarray] = []
    path_vertex_ids: list[int] = []

    n_nodes = len(nodes)
    for i in range(n_nodes):
        src_face = int(tri_ids[i])
        dst_face = int(tri_ids[(i + 1) % n_nodes])
        src_b = bary_coords[i]
        dst_b = bary_coords[(i + 1) % n_nodes]

        source_sp = geodesic_mesh.make_face_point(
            src_face, float(src_b[0]), float(src_b[1]), float(src_b[2])
        )
        target_sp = geodesic_mesh.make_face_point(
            dst_face, float(dst_b[0]), float(dst_b[1]), float(dst_b[2])
        )

        segment = geodesic_mesh.trace_path_points(source_sp, target_sp)
        for j, sp in enumerate(segment):
            if i > 0 and j == 0:
                continue
            path_points.append(
                _surface_point_to_xyz(sp, gc_vertices, gc_edges, gc_faces)
            )
            path_vertex_ids.append(int(sp.index) if sp.type == 0 else -1)

    points_arr = np.asarray(path_points, dtype=np.float64)
    vertex_ids_arr = np.asarray(path_vertex_ids, dtype=np.int64)
    return GeodesicCurvePath(points=points_arr, vertex_indices=vertex_ids_arr)
