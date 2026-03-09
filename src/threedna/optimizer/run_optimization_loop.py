import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from threedna import paths
from threedna.init_curve.initialize_ring import initialize_ring_on_surface
from threedna.mesh_io.model_loader import load_3d_model
from threedna.optimizer.medial_axis_energy import optimize_curve_newton
from threedna.viz.visualize_curve_on_surface import render_curve_on_surface


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal Newton-style curve optimization loop"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=(
            paths.PROJECT_ROOT
            / "src/threedna/assets/models/simple_capsule_basic_watertight.glb"
        ),
        help="Path to watertight mesh",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=64,
        help="Number of points in the initialized ring",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=3,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Weight for the medial-axis term",
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=0.05,
        help="Weight for the length regularizer",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        help="Newton update step size in (0, 1]",
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
    args = parser.parse_args()

    mesh = load_3d_model(
        args.mesh,
        scale_axis=args.scale_axis,
        target_length_nm=args.target_length_nm,
    )
    print("Surface projection backend: C++")

    initial_curve = initialize_ring_on_surface(
        mesh=mesh, n_points=args.n_points
    )
    result = optimize_curve_newton(
        mesh=mesh,
        initial_curve=initial_curve,
        alpha=args.alpha,
        length_weight=args.length_weight,
        n_iters=args.n_iters,
        step_size=args.step_size,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths.PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_path = render_curve_on_surface(
        mesh=mesh,
        curve=initial_curve,
        output_glb=output_dir / "initial_curve.glb",
    )
    optimized_curve_3xn = result.curve.T.astype(np.float32)
    optimized_path = render_curve_on_surface(
        mesh=mesh,
        curve=optimized_curve_3xn,
        output_glb=output_dir / "optimized_curve.glb",
    )
    np.save(output_dir / "initial_curve.npy", initial_curve)
    np.save(output_dir / "optimized_curve.npy", optimized_curve_3xn)
    np.save(output_dir / "energies.npy", result.energies)

    print(f"Saved initial curve render: {initial_path}")
    print(f"Saved optimized curve render: {optimized_path}")
    print(f"Saved optimization energies: {output_dir / 'energies.npy'}")
    print("Energies per iteration:")
    for idx, value in enumerate(result.energies, start=1):
        print(f"  iter {idx}: {value:.6f}")


if __name__ == "__main__":
    main()
