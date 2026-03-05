from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClosedCurveParam:
    center: np.ndarray
    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray


def _as_points(curve: np.ndarray) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("curve must be a 2D array")
    if arr.shape[0] == 3:
        points = arr.T
    elif arr.shape[1] == 3:
        points = arr
    else:
        raise ValueError("curve must have shape (3, n_points) or (n_points, 3)")
    if len(points) < 8:
        raise ValueError("curve must contain at least 8 points")
    if np.linalg.norm(points[0] - points[-1]) <= 1e-12:
        points = points[:-1]
    return points


def fit_closed_curve_parametrization(
    curve: np.ndarray,
    *,
    n_harmonics: int = 8,
) -> ClosedCurveParam:
    """
    Fit a smooth closed curve using a truncated Fourier basis.

    The returned parametrization is continuous and periodic over t in [0, 1).
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")

    points = _as_points(curve)
    n_points = len(points)
    if n_harmonics >= n_points // 2:
        raise ValueError("n_harmonics is too high for the number of samples")

    t = 2.0 * np.pi * (np.arange(n_points, dtype=np.float64) / n_points)
    cols = [np.ones(n_points, dtype=np.float64)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * t))
        cols.append(np.sin(k * t))
    design = np.stack(cols, axis=1)

    coeffs, _, _, _ = np.linalg.lstsq(design, points, rcond=None)
    center = coeffs[0]
    cos_coeffs = coeffs[1::2]
    sin_coeffs = coeffs[2::2]
    return ClosedCurveParam(center=center, cos_coeffs=cos_coeffs, sin_coeffs=sin_coeffs)


def sample_closed_curve(
    param: ClosedCurveParam,
    *,
    n_points: int,
) -> np.ndarray:
    """
    Sample a fitted closed curve and return shape (3, n_points).
    """
    if n_points < 8:
        raise ValueError("n_points must be >= 8")

    t = 2.0 * np.pi * (np.arange(n_points, dtype=np.float64) / n_points)
    points = np.tile(param.center[None, :], (n_points, 1))
    for idx in range(len(param.cos_coeffs)):
        k = idx + 1
        points += np.cos(k * t)[:, None] * param.cos_coeffs[idx][None, :]
        points += np.sin(k * t)[:, None] * param.sin_coeffs[idx][None, :]

    return points.T.astype(np.float32)


def discrete_curvature(points_3xn: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Discrete curvature proxy at each sample, for regularization.
    """
    points = _as_points(points_3xn)
    prev = np.roll(points, 1, axis=0)
    nxt = np.roll(points, -1, axis=0)

    e_prev = points - prev
    e_next = nxt - points

    l_prev = np.linalg.norm(e_prev, axis=1)
    l_next = np.linalg.norm(e_next, axis=1)

    t_prev = e_prev / (l_prev[:, None] + eps)
    t_next = e_next / (l_next[:, None] + eps)
    ds = 0.5 * (l_prev + l_next) + eps
    return np.linalg.norm(t_next - t_prev, axis=1) / ds
