import numpy as np
import trimesh

from threedna.optimizer.implicit_medial_axis import compute_implicit_medial_axis


def test_compute_implicit_medial_axis_shapes_and_finiteness() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    n = 24
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    curve = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(
        np.float32
    )

    result = compute_implicit_medial_axis(
        mesh,
        curve,
        tolerance=1e-3,
        max_expand_steps=8,
        max_binary_steps=12,
    )

    assert result.m_plus.shape == (n, 3)
    assert result.m_minus.shape == (n, 3)
    assert result.r_plus.shape == (n,)
    assert result.r_minus.shape == (n,)
    assert result.plus_witness_index.shape == (n,)
    assert result.minus_witness_index.shape == (n,)

    assert np.all(np.isfinite(result.m_plus))
    assert np.all(np.isfinite(result.m_minus))
    assert np.all(np.isfinite(result.r_plus))
    assert np.all(np.isfinite(result.r_minus))
    assert np.all(result.r_plus >= 0.0)
    assert np.all(result.r_minus >= 0.0)
