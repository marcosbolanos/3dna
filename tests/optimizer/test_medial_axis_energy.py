import numpy as np
from scipy import sparse
import trimesh

from threedna.optimizer.medial_axis_energy import (
    build_length_quadratic,
    build_medial_axis_quadratic,
    compute_medial_axis_energy,
    evaluate_quadratic_form,
    medial_axis_energy,
    optimize_curve_newton,
)


def test_medial_axis_quadratic_matches_direct_energy() -> None:
    n = 20
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    curve = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

    offset = np.array([0.0, 0.0, 0.2], dtype=np.float64)
    m_plus = curve + offset[None, :]
    m_minus = curve - offset[None, :]

    form = build_medial_axis_quadratic(curve, m_plus, m_minus)
    x = curve.reshape(-1)

    direct = medial_axis_energy(curve, m_plus, m_minus)
    quad = evaluate_quadratic_form(form, x)
    assert np.isclose(direct, quad)
    assert sparse.isspmatrix_csr(form.matrix)
    assert form.matrix.nnz == 3 * n


def test_compute_medial_axis_energy_returns_consistent_values() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    n = 20
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    curve = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(
        np.float32
    )

    result = compute_medial_axis_energy(
        mesh,
        curve,
        tolerance=1e-3,
        max_expand_steps=8,
        max_binary_steps=16,
    )

    assert np.isfinite(result.energy)
    assert np.isfinite(result.direct_energy)
    assert np.isclose(result.energy, result.direct_energy, rtol=1e-5, atol=1e-6)

    dim = 3 * n
    assert result.quadratic.matrix.shape == (dim, dim)
    assert result.quadratic.linear.shape == (dim,)
    assert (result.quadratic.matrix - result.quadratic.matrix.T).nnz == 0


def test_build_length_quadratic_is_sparse_symmetric() -> None:
    n = 16
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    curve = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

    A, linear = build_length_quadratic(curve, length_weight=0.5)

    assert sparse.isspmatrix_csr(A)
    assert A.shape == (3 * n, 3 * n)
    assert linear.shape == (3 * n,)
    assert np.allclose(linear, 0.0)
    assert (A - A.T).nnz == 0


def test_optimize_curve_newton_runs() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    n = 16
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    init_curve = np.vstack(
        [0.9 * np.cos(theta), 0.9 * np.sin(theta), np.zeros_like(theta)]
    )

    result = optimize_curve_newton(
        mesh,
        init_curve,
        alpha=0.2,
        length_weight=0.05,
        n_iters=2,
        tolerance=1e-3,
        max_expand_steps=6,
        max_binary_steps=10,
    )

    assert result.curve.shape == (n, 3)
    assert result.energies.shape == (2,)
    assert np.all(np.isfinite(result.curve))
    assert np.all(np.isfinite(result.energies))
