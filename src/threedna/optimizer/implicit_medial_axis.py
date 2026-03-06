from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class ClosestPointQuery:
    points: np.ndarray
    curve_indices: np.ndarray
    kdtree: Any


@dataclass(frozen=True)
class ImplicitMedialAxisResult:
    m_plus: np.ndarray
    m_minus: np.ndarray
    r_plus: np.ndarray
    r_minus: np.ndarray
    plus_witness_index: np.ndarray
    minus_witness_index: np.ndarray


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
    if len(points) < 8:
        raise ValueError("curve must contain at least 8 points")
    return points


def _boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    edges = np.vstack([e01, e12, e20])
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return np.empty((0, 3), dtype=np.float64)
    boundary_idx = np.unique(boundary_edges.reshape(-1))
    return np.asarray(mesh.vertices, dtype=np.float64)[boundary_idx]


def _build_closest_point_query(
    mesh: trimesh.Trimesh, curve_points: np.ndarray
) -> ClosestPointQuery:
    boundary = _boundary_vertices(mesh)
    n_curve = len(curve_points)

    if len(boundary) == 0:
        points = curve_points.copy()
        curve_indices = np.arange(n_curve, dtype=np.int64)
    else:
        points = np.vstack([curve_points, boundary])
        curve_indices = np.full(len(points), -1, dtype=np.int64)
        curve_indices[:n_curve] = np.arange(n_curve, dtype=np.int64)

    kdtree = trimesh.points.PointCloud(points).kdtree
    return ClosestPointQuery(points=points, curve_indices=curve_indices, kdtree=kdtree)


def _query_nearest(
    cpq: ClosestPointQuery,
    center: np.ndarray,
    *,
    curve_node_idx: int,
    n_curve: int,
    exclude_neighbor_hops: int,
) -> tuple[float, int]:
    n_points = len(cpq.points)
    if n_points == 0:
        return np.inf, -1

    def is_valid(candidate_idx: int) -> bool:
        if exclude_neighbor_hops < 0:
            return True
        curve_j = int(cpq.curve_indices[candidate_idx])
        if curve_j < 0:
            return True
        cyclic_diff = abs(curve_j - curve_node_idx)
        cyclic_diff = min(cyclic_diff, n_curve - cyclic_diff)
        return cyclic_diff > exclude_neighbor_hops

    k = 8
    while True:
        k_eff = min(k, n_points)
        dists, idxs = cpq.kdtree.query(center, k=k_eff)

        dists_arr = np.atleast_1d(np.asarray(dists, dtype=np.float64))
        idxs_arr = np.atleast_1d(np.asarray(idxs, dtype=np.int64))

        for dist, idx in zip(dists_arr, idxs_arr, strict=False):
            if idx < 0 or idx >= n_points:
                continue
            if not np.isfinite(dist):
                continue
            if is_valid(int(idx)):
                return float(dist), int(idx)

        if k_eff >= n_points:
            return np.inf, -1
        k *= 2


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _curve_binormals(mesh: trimesh.Trimesh, curve_points: np.ndarray) -> np.ndarray:
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    _, nearest_vertex = mesh.kdtree.query(curve_points)
    surf_normals = vertex_normals[np.asarray(nearest_vertex, dtype=np.int64)]

    n = len(curve_points)
    binormals = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        prev_pt = curve_points[(i - 1) % n]
        curr_pt = curve_points[i]
        next_pt = curve_points[(i + 1) % n]

        tangent = _normalize(next_pt - prev_pt)
        normal = _normalize(surf_normals[i])
        b = np.cross(tangent, normal)
        b = _normalize(b)
        if np.linalg.norm(b) <= 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(tangent, axis))) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            b = _normalize(np.cross(tangent, axis))
        if np.linalg.norm(b) <= 1e-12:
            b = _normalize(curr_pt - np.asarray(mesh.center_mass, dtype=np.float64))
        binormals[i] = b
    return binormals


def _search_side(
    cpq: ClosestPointQuery,
    curve_points: np.ndarray,
    i: int,
    direction: np.ndarray,
    *,
    exclude_neighbor_hops: int,
    tolerance: float,
    initial_radius: float,
    max_expand_steps: int,
    max_binary_steps: int,
) -> tuple[np.ndarray, float, int]:
    n_curve = len(curve_points)
    gamma_i = curve_points[i]
    direction = _normalize(direction)
    if np.linalg.norm(direction) <= 1e-12:
        return gamma_i.copy(), 0.0, -1

    def signed_gap(radius: float) -> tuple[float, float, int]:
        center = gamma_i + radius * direction
        d_near, idx = _query_nearest(
            cpq,
            center,
            curve_node_idx=i,
            n_curve=n_curve,
            exclude_neighbor_hops=exclude_neighbor_hops,
        )
        return d_near - radius, d_near, idx

    lo = 0.0
    hi = max(initial_radius, tolerance)
    crossed = False
    witness_idx = -1

    for _ in range(max_expand_steps):
        gap_hi, _, idx_hi = signed_gap(hi)
        witness_idx = idx_hi
        if gap_hi <= 0.0:
            crossed = True
            break
        lo = hi
        hi *= 2.0

    if crossed:
        for _ in range(max_binary_steps):
            mid = 0.5 * (lo + hi)
            gap_mid, _, idx_mid = signed_gap(mid)
            witness_idx = idx_mid
            if gap_mid > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tolerance:
                break
        radius = 0.5 * (lo + hi)
    else:
        radius = hi

    center = gamma_i + radius * direction
    _, witness_idx = _query_nearest(
        cpq,
        center,
        curve_node_idx=i,
        n_curve=n_curve,
        exclude_neighbor_hops=exclude_neighbor_hops,
    )
    return center, float(radius), witness_idx


def compute_implicit_medial_axis(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
    *,
    exclude_neighbor_hops: int = 1,
    tolerance: float = 1e-4,
    max_expand_steps: int = 16,
    max_binary_steps: int = 32,
    initial_radius_scale: float = 1.5,
) -> ImplicitMedialAxisResult:
    """
    Compute implicit medial axis centers m+ and m- for each discretized curve node.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")
    if max_expand_steps < 1:
        raise ValueError("max_expand_steps must be >= 1")
    if max_binary_steps < 1:
        raise ValueError("max_binary_steps must be >= 1")
    if initial_radius_scale <= 0.0:
        raise ValueError("initial_radius_scale must be > 0")

    curve_points = _as_curve_points(curve)
    n = len(curve_points)
    cpq = _build_closest_point_query(mesh, curve_points)
    binormals = _curve_binormals(mesh, curve_points)

    edge_len = np.linalg.norm(np.roll(curve_points, -1, axis=0) - curve_points, axis=1)
    median_edge = float(np.median(edge_len))
    base_radius = max(initial_radius_scale * median_edge, tolerance)

    m_plus = np.empty((n, 3), dtype=np.float64)
    m_minus = np.empty((n, 3), dtype=np.float64)
    r_plus = np.empty(n, dtype=np.float64)
    r_minus = np.empty(n, dtype=np.float64)
    plus_idx = np.empty(n, dtype=np.int64)
    minus_idx = np.empty(n, dtype=np.int64)

    for i in range(n):
        c_plus, rad_plus, idx_plus = _search_side(
            cpq,
            curve_points,
            i,
            binormals[i],
            exclude_neighbor_hops=exclude_neighbor_hops,
            tolerance=tolerance,
            initial_radius=base_radius,
            max_expand_steps=max_expand_steps,
            max_binary_steps=max_binary_steps,
        )
        c_minus, rad_minus, idx_minus = _search_side(
            cpq,
            curve_points,
            i,
            -binormals[i],
            exclude_neighbor_hops=exclude_neighbor_hops,
            tolerance=tolerance,
            initial_radius=base_radius,
            max_expand_steps=max_expand_steps,
            max_binary_steps=max_binary_steps,
        )
        m_plus[i] = c_plus
        m_minus[i] = c_minus
        r_plus[i] = rad_plus
        r_minus[i] = rad_minus
        plus_idx[i] = idx_plus
        minus_idx[i] = idx_minus

    return ImplicitMedialAxisResult(
        m_plus=m_plus,
        m_minus=m_minus,
        r_plus=r_plus,
        r_minus=r_minus,
        plus_witness_index=plus_idx,
        minus_witness_index=minus_idx,
    )
