from pathlib import Path
import sys

import trimesh


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
        print(
            "\033[31mGiven mesh is not watertight. Provide a watertight model "
            "or use the included watertight helper tool.\033[0m",
            file=sys.stderr,
        )
    return mesh
