from __future__ import annotations

import numpy

__all__: list[str] = ["project_points_to_mesh"]

def project_points_to_mesh(
    vertices: numpy.ndarray,
    faces: numpy.ndarray,
    points: numpy.ndarray,
) -> numpy.ndarray: ...
