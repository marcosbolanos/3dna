import numpy as np
import pytest
import trimesh

from threedna.init_curve.initialize_ring import initialize_ring_on_surface


def test_initialize_ring_on_watertight_sphere() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    ring = initialize_ring_on_surface(mesh, n_points=64)

    assert ring.shape == (3, 64)
    assert ring.dtype == np.float32
    radii = np.linalg.norm(ring.T, axis=1)
    assert np.allclose(radii, 1.0, atol=0.12)


def test_initialize_ring_requires_watertight_mesh() -> None:
    open_mesh = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )

    with pytest.raises(ValueError, match="watertight"):
        initialize_ring_on_surface(open_mesh)
