import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

DEFAULT_SAMPLE_POINTS = 250000
DEFAULT_POISSON_DEPTH = 9
DEFAULT_DENSITY_TRIM_QUANTILE = 0.0
DEFAULT_TARGET_FACES = 15000


def make_watertight_open3d(
    mesh: trimesh.Trimesh,
    *,
    sample_points: int = DEFAULT_SAMPLE_POINTS,
    poisson_depth: int = DEFAULT_POISSON_DEPTH,
    density_trim_quantile: float = DEFAULT_DENSITY_TRIM_QUANTILE,
    target_faces: int = DEFAULT_TARGET_FACES,
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


def _load_single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise ValueError(f"no mesh geometry found in {path}")
        if len(geoms) > 1:
            raise ValueError(
                "input contains multiple mesh geometries; provide a single mesh"
            )
        mesh = geoms[0].copy()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise ValueError(f"unsupported mesh payload in {path}: {type(loaded).__name__}")

    mesh.remove_unreferenced_vertices()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a mesh watertight with Open3D")
    parser.add_argument("--input", type=Path, required=True, help="Input mesh path")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh path")
    parser.add_argument(
        "--sample-points",
        type=int,
        default=DEFAULT_SAMPLE_POINTS,
        help=f"Poisson disk sample points (default: {DEFAULT_SAMPLE_POINTS})",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=DEFAULT_POISSON_DEPTH,
        help=f"Poisson reconstruction depth (default: {DEFAULT_POISSON_DEPTH})",
    )
    parser.add_argument(
        "--density-trim-quantile",
        type=float,
        default=DEFAULT_DENSITY_TRIM_QUANTILE,
        help=(
            "Trim vertices below this density quantile in [0.0, 1.0) "
            f"(default: {DEFAULT_DENSITY_TRIM_QUANTILE})"
        ),
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=DEFAULT_TARGET_FACES,
        help=f"Target triangle count after decimation (default: {DEFAULT_TARGET_FACES})",
    )
    args = parser.parse_args()

    mesh = _load_single_mesh(args.input)
    out = make_watertight_open3d(
        mesh,
        sample_points=args.sample_points,
        poisson_depth=args.poisson_depth,
        density_trim_quantile=args.density_trim_quantile,
        target_faces=args.target_faces,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.export(args.output)


if __name__ == "__main__":
    main()
