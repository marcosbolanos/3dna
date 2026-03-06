import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.color import ColorVisuals

from threedna.geodesic_curve import reconstruct_geodesic_curve_on_mesh
from threedna import paths
from threedna.init_curve.initialize_ring import initialize_ring_on_surface
from threedna.mesh_io.model_loader import load_3d_model


def _curve_points(curve: np.ndarray) -> np.ndarray:
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


def build_curve_on_surface_scene(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
    *,
    geodesic_reconstruction: bool = True,
) -> trimesh.Scene:
    points = _curve_points(curve)
    if geodesic_reconstruction:
        geodesic_path = reconstruct_geodesic_curve_on_mesh(mesh, points)
        closed = geodesic_path.points
    else:
        closed = np.vstack([points, points[0]])

    mesh_geom = mesh.copy()
    mesh_geom.visual = ColorVisuals(
        mesh=mesh_geom,
        face_colors=np.array([170, 180, 190, 180], dtype=np.uint8),
    )

    curve_path = trimesh.load_path(closed)

    scene = trimesh.Scene()
    scene.add_geometry(mesh_geom, geom_name="surface_mesh")
    scene.add_geometry(curve_path, geom_name="initialized_ring")
    return scene


def render_curve_on_surface(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
    output_glb: Path,
    *,
    geodesic_reconstruction: bool = True,
) -> Path:
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene = build_curve_on_surface_scene(
        mesh=mesh,
        curve=curve,
        geodesic_reconstruction=geodesic_reconstruction,
    )
    scene.export(output_glb)
    return output_glb


def main() -> None:
    parser = argparse.ArgumentParser(description="Render initialized ring on a mesh")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=(
            paths.PROJECT_ROOT
            / "src/threedna/assets/models/simple_capsule_basic_watertight.glb"
        ),
        help="Path to watertight mesh (.glb)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=128,
        help="Number of sampled points in initialized ring",
    )
    parser.add_argument(
        "--scale-axis",
        type=str,
        default=None,
        help="Optional axis for scale calibration (x, y, z)",
    )
    parser.add_argument(
        "--target-length-nm",
        type=float,
        default=None,
        help="Optional target size along scale-axis in nm",
    )
    parser.add_argument(
        "--straight-segments",
        action="store_true",
        help="Disable geodesic segment reconstruction for rendering",
    )
    args = parser.parse_args()

    mesh = load_3d_model(
        args.mesh,
        scale_axis=args.scale_axis,
        target_length_nm=args.target_length_nm,
    )
    ring = initialize_ring_on_surface(mesh=mesh, n_points=args.n_points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths.PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_path = render_curve_on_surface(
        mesh=mesh,
        curve=ring,
        output_glb=output_dir / "curve_on_surface.glb",
        geodesic_reconstruction=not args.straight_segments,
    )
    np.save(output_dir / "initialized_ring.npy", ring)

    print(f"Saved visualization to {glb_path}")
    print(f"Saved ring points to {output_dir / 'initialized_ring.npy'}")


if __name__ == "__main__":
    main()
