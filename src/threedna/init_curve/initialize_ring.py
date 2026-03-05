import numpy as np
import trimesh


def initialize_ring_on_surface(
    mesh: trimesh.Trimesh,
    n_points: int = 128,
) -> np.ndarray:
    """
    Initialize a closed ring curve by intersecting a watertight mesh with a plane.

    The plane passes through the mesh center and uses the principal-axis orientation
    that yields the longest valid closed section.
    """
    if n_points < 8:
        raise ValueError("n_points must be >= 8")
    if not mesh.is_watertight:
        raise ValueError(
            "initialize_ring_on_surface requires a watertight mesh. "
            "Provide a watertight model or use threedna.mesh_io.watertight_helper."
        )

    center = np.asarray(mesh.center_mass, dtype=np.float64)
    if not np.isfinite(center).all():
        center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)

    verts = np.asarray(mesh.vertices, dtype=np.float64) - center
    cov = verts.T @ verts / max(len(verts) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)

    def key(point: np.ndarray, tol: float = 1e-6) -> tuple[int, int, int]:
        q = np.round(point / tol).astype(np.int64)
        return int(q[0]), int(q[1]), int(q[2])

    def loops_from_segments(segments: np.ndarray) -> list[np.ndarray]:
        if segments.size == 0:
            return []

        vertices: dict[tuple[int, int, int], np.ndarray] = {}
        adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

        for seg in segments:
            a = np.asarray(seg[0], dtype=np.float64)
            b = np.asarray(seg[1], dtype=np.float64)
            if np.linalg.norm(a - b) <= 1e-12:
                continue
            ka = key(a)
            kb = key(b)
            vertices.setdefault(ka, a)
            vertices.setdefault(kb, b)
            adjacency.setdefault(ka, []).append(kb)
            adjacency.setdefault(kb, []).append(ka)

        def edge_key(
            a: tuple[int, int, int], b: tuple[int, int, int]
        ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
            return (a, b) if a <= b else (b, a)

        loops: list[np.ndarray] = []
        visited_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

        for start, neighbors in adjacency.items():
            for nxt in neighbors:
                edge = edge_key(start, nxt)
                if edge in visited_edges:
                    continue

                visited_edges.add(edge)
                chain = [start, nxt]
                prev = start
                curr = nxt

                while True:
                    curr_neighbors = adjacency.get(curr, [])
                    candidates = [node for node in curr_neighbors if node != prev]
                    if not candidates:
                        break
                    next_node = candidates[0]
                    next_edge = edge_key(curr, next_node)
                    if next_edge in visited_edges:
                        break
                    visited_edges.add(next_edge)
                    chain.append(next_node)
                    prev, curr = curr, next_node
                    if curr == start:
                        break

                if len(chain) >= 4 and chain[-1] == start:
                    loop_points = np.array(
                        [vertices[node] for node in chain], dtype=np.float64
                    )
                    loops.append(loop_points)

        return loops

    best_loop: np.ndarray | None = None
    best_length = -1.0
    for axis_idx in range(3):
        normal = eigvecs[:, axis_idx]
        segments = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=normal,
            plane_origin=center,
        )
        if len(segments) == 0:
            continue

        for closed in loops_from_segments(np.asarray(segments, dtype=np.float64)):
            segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            loop_length = float(segment_lengths.sum())
            if loop_length > best_length:
                best_length = loop_length
                best_loop = closed

    if best_loop is None:
        raise ValueError("Failed to compute a valid closed section from the mesh")

    segment_lengths = np.linalg.norm(np.diff(best_loop, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0.0:
        raise ValueError("Section loop has zero length")

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_distances = np.linspace(0.0, total_length, num=n_points, endpoint=False)
    ring = np.empty((n_points, 3), dtype=np.float64)

    for i, dist in enumerate(sample_distances):
        seg_idx = int(np.searchsorted(cumulative, dist, side="right") - 1)
        seg_idx = min(seg_idx, len(segment_lengths) - 1)
        seg_len = segment_lengths[seg_idx]
        if seg_len == 0.0:
            ring[i] = best_loop[seg_idx]
            continue
        t = (dist - cumulative[seg_idx]) / seg_len
        ring[i] = (1.0 - t) * best_loop[seg_idx] + t * best_loop[seg_idx + 1]

    return ring.T.astype(np.float32)
