from pathlib import Path
import trimesh
import open3d as o3d
import numpy as np


def _make_watertight_open3d(
    mesh: trimesh.Trimesh,
    *,
    sample_points: int,
    poisson_depth: int,
    density_trim_quantile: float,
    target_faces: int,
) -> trimesh.Trimesh:

    if sample_points < 1000:
        raise ValueError("sample_points must be >= 1000")
    if poisson_depth < 5:
        raise ValueError("poisson_depth must be >= 5")
    if not 0.0 <= density_trim_quantile < 1.0:
        raise ValueError("density_trim_quantile must be in [0.0, 1.0)")
    if target_faces < 0:
        raise ValueError("target_faces must be >= 0")

    src = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
    )
    src.compute_vertex_normals()

    pcd = src.sample_points_poisson_disk(number_of_points=sample_points)
    pcd.estimate_normals()

    rec, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth,
    )
    dens = np.asarray(densities)
    if dens.size == 0:
        raise ValueError("Open3D Poisson reconstruction produced empty density data")

    if density_trim_quantile > 0.0:
        threshold = float(np.quantile(dens, density_trim_quantile))
        rec.remove_vertices_by_mask(dens < threshold)

    rec.remove_duplicated_vertices()
    rec.remove_duplicated_triangles()
    rec.remove_degenerate_triangles()
    rec.remove_non_manifold_edges()
    rec.remove_unreferenced_vertices()

    if target_faces > 0 and len(rec.triangles) > target_faces:
        rec = rec.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        rec.remove_degenerate_triangles()
        rec.remove_non_manifold_edges()
        rec.remove_unreferenced_vertices()

    verts = np.asarray(rec.vertices)
    faces = np.asarray(rec.triangles)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("Open3D reconstruction produced an empty mesh")

    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.remove_unreferenced_vertices()
    return out


def load_3d_model(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise ValueError(f"no mesh geometry found in {path}")
        if len(geoms) > 1:
            raise ValueError(
                "input contains multiple mesh geometries; provide a single closed mesh "
            )
        mesh = geoms[0].copy()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise ValueError(f"unsupported mesh payload in {path}: {type(loaded).__name__}")

    mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        mesh = _make_watertight_open3d(
            mesh=mesh,
            sample_points=250000,
            poisson_depth=9,
            density_trim_quantile=0.0,
            target_faces=15000,
        )
    return mesh
