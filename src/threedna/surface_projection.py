import numpy as np
import trimesh

try:
    import threedna._surface_kernels_cpp as _surface_kernels_cpp  # pyright: ignore[reportMissingImports]
except ImportError:
    _surface_kernels_cpp = None


def using_cpp_backend() -> bool:
    return _surface_kernels_cpp is not None


def _closest_point_on_triangles(triangles: np.ndarray, point: np.ndarray) -> np.ndarray:
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    ab = b - a
    ac = c - a
    ap = point[None, :] - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    out = np.empty_like(a)
    done = np.zeros(len(triangles), dtype=bool)

    vc0 = (d1 <= 0.0) & (d2 <= 0.0)
    if np.any(vc0):
        out[vc0] = a[vc0]
        done[vc0] = True

    bp = point[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    vc1 = (~done) & (d3 >= 0.0) & (d4 <= d3)
    if np.any(vc1):
        out[vc1] = b[vc1]
        done[vc1] = True

    cp = point[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc2 = (~done) & (d6 >= 0.0) & (d5 <= d6)
    if np.any(vc2):
        out[vc2] = c[vc2]
        done[vc2] = True

    vc = d1 * d4 - d3 * d2
    vc3 = (~done) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(vc3):
        v = d1[vc3] / (d1[vc3] - d3[vc3])
        out[vc3] = a[vc3] + v[:, None] * ab[vc3]
        done[vc3] = True

    vb = d5 * d2 - d1 * d6
    vc4 = (~done) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(vc4):
        w = d2[vc4] / (d2[vc4] - d6[vc4])
        out[vc4] = a[vc4] + w[:, None] * ac[vc4]
        done[vc4] = True

    va = d3 * d6 - d5 * d4
    vc5 = (~done) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & (va <= 0.0)
    if np.any(vc5):
        denom = (d4[vc5] - d3[vc5]) + (d5[vc5] - d6[vc5])
        w = (d4[vc5] - d3[vc5]) / denom
        out[vc5] = b[vc5] + w[:, None] * (c[vc5] - b[vc5])
        done[vc5] = True

    vc6 = ~done
    if np.any(vc6):
        denom = va[vc6] + vb[vc6] + vc[vc6]
        inv = 1.0 / denom
        v = vb[vc6] * inv
        w = vc[vc6] * inv
        out[vc6] = a[vc6] + ab[vc6] * v[:, None] + ac[vc6] * w[:, None]

    diff = out - point[None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    return out[int(np.argmin(dist2))]


def _project_points_python(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    triangles = vertices[faces]
    projected = np.empty_like(points)
    for i in range(len(points)):
        projected[i] = _closest_point_on_triangles(triangles, points[i])
    return projected


def project_points_to_mesh(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("mesh contains no triangles")

    if _surface_kernels_cpp is not None:
        projected = _surface_kernels_cpp.project_points_to_mesh(vertices, faces, query)
        return np.asarray(projected, dtype=np.float64)

    return _project_points_python(vertices, faces, query)
