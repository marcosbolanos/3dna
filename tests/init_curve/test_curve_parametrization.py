import numpy as np

from threedna.init_curve.curve_parametrization import (
    discrete_curvature,
    fit_closed_curve_parametrization,
    sample_closed_curve,
)


def test_fit_and_sample_closed_curve_circle() -> None:
    n = 128
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(
        np.float32
    )

    param = fit_closed_curve_parametrization(circle, n_harmonics=3)
    sampled = sample_closed_curve(param, n_points=n)

    assert sampled.shape == (3, n)
    radii = np.linalg.norm(sampled.T[:, :2], axis=1)
    assert np.allclose(radii, 1.0, atol=1e-2)


def test_discrete_curvature_is_nearly_constant_for_circle() -> None:
    n = 128
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(
        np.float32
    )

    kappa = discrete_curvature(circle)

    assert kappa.shape == (n,)
    assert np.std(kappa) < 0.05
