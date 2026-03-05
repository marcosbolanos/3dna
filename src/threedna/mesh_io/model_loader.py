from pathlib import Path
import sys

import numpy as np
import trimesh


def load_3d_model(
    path: Path,
    *,
    scale_axis: str | None = None,
    target_length_nm: float | None = None,
) -> trimesh.Trimesh:
    """
    Load a 3D model, verify that the model is watertight, and scale it to desired scale
    params:

    path: Path object pointing to the model you want to load
    scale_axis: axis along which scaling will be done
    target_length_nm: desired length of the specified axis in nm
    """
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

    if (scale_axis is None) != (target_length_nm is None):
        raise ValueError("scale_axis and target_length_nm must be provided together")

    if scale_axis is not None and target_length_nm is not None:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        axis = scale_axis.lower()
        if axis not in axis_to_index:
            raise ValueError("scale_axis must be one of: x, y, z")
        if target_length_nm <= 0.0:
            raise ValueError("target_length_nm must be > 0")

        vertices = np.asarray(mesh.vertices)
        axis_values = vertices[:, axis_to_index[axis]]
        current_length = float(axis_values.max() - axis_values.min())
        if current_length <= 0.0:
            raise ValueError(
                f"mesh has zero extent along axis '{axis}', cannot calibrate scale"
            )

        mesh.apply_scale(target_length_nm / current_length)

    if not mesh.is_watertight:
        print(
            "\033[31mGiven mesh is not watertight. Provide a watertight model "
            "or use the included watertight helper tool.\033[0m",
            file=sys.stderr,
        )
    return mesh
