from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import trimesh

from threedna.optimizer.implicit_medial_axis import (
    ImplicitMedialAxisResult,
    compute_implicit_medial_axis,
)
from threedna.surface_projection import project_points_to_mesh


@dataclass(frozen=True)
class QuadraticForm:
    matrix: Any
    linear: np.ndarray
    constant: float


@dataclass(frozen=True)
class MedialAxisEnergyResult:
    energy: float
    direct_energy: float
    quadratic: QuadraticForm
    implicit_axis: ImplicitMedialAxisResult
    weights: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    curve: np.ndarray
    energies: np.ndarray


def _as_points_3d(name: str, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[0] == 3:
        points = arr.T
    elif arr.shape[1] == 3:
        points = arr
    else:
        raise ValueError(
            f"{name} must have shape (3, n_points) or (n_points, 3)"
        )
    return points


def vertex_integration_weights(curve: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex quadrature weights (l_{i-1} + l_i) / 2.
    """
    points = _as_points_3d("curve", curve)
    if len(points) < 2:
        raise ValueError("curve must contain at least 2 points")
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return 0.5 * (np.roll(edge_lengths, 1) + edge_lengths)


def medial_axis_energy(
    curve: np.ndarray,
    m_plus: np.ndarray,
    m_minus: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """
    Evaluate the discretized medial axis energy from Eq. (7).
    """
    gamma = _as_points_3d("curve", curve)
    plus = _as_points_3d("m_plus", m_plus)
    minus = _as_points_3d("m_minus", m_minus)

    if len(gamma) != len(plus) or len(gamma) != len(minus):
        raise ValueError(
            "curve, m_plus, and m_minus must have the same number of nodes"
        )

    w = (
        vertex_integration_weights(gamma)
        if weights is None
        else np.asarray(weights)
    )
    if w.shape != (len(gamma),):
        raise ValueError("weights must have shape (n_points,)")

    plus_term = np.sum((gamma - plus) ** 2, axis=1)
    minus_term = np.sum((gamma - minus) ** 2, axis=1)
    return float(np.sum(w * (plus_term + minus_term)))


def build_medial_axis_quadratic(
    curve: np.ndarray,
    m_plus: np.ndarray,
    m_minus: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> QuadraticForm:
    """
    Build Eq. (8): E_M(gamma) = 0.5 * gamma^T M gamma + n^T gamma + c.
    """
    gamma = _as_points_3d("curve", curve)
    plus = _as_points_3d("m_plus", m_plus)
    minus = _as_points_3d("m_minus", m_minus)

    if len(gamma) != len(plus) or len(gamma) != len(minus):
        raise ValueError(
            "curve, m_plus, and m_minus must have the same number of nodes"
        )

    w = (
        vertex_integration_weights(gamma)
        if weights is None
        else np.asarray(weights)
    )
    if w.shape != (len(gamma),):
        raise ValueError("weights must have shape (n_points,)")

    diagonal = np.repeat(4.0 * w, 3)
    matrix = sparse.csr_matrix(sparse.diags(diagonal, offsets=0, format="csr"))

    linear_points = -2.0 * w[:, None] * (plus + minus)
    linear = linear_points.reshape(-1)

    const_plus = np.sum(plus**2, axis=1)
    const_minus = np.sum(minus**2, axis=1)
    constant = float(np.sum(w * (const_plus + const_minus)))

    return QuadraticForm(matrix=matrix, linear=linear, constant=constant)


def evaluate_quadratic_form(form: QuadraticForm, x: np.ndarray) -> float:
    vec = np.asarray(x, dtype=np.float64).reshape(-1)
    matrix_shape = form.matrix.shape
    if matrix_shape is None:
        raise ValueError("form.matrix shape is undefined")
    n_rows, n_cols = int(matrix_shape[0]), int(matrix_shape[1])
    if (n_rows, n_cols) != (len(vec), len(vec)):
        raise ValueError("form.matrix shape does not match x")
    if form.linear.shape != (len(vec),):
        raise ValueError("form.linear shape does not match x")
    quadratic = 0.5 * float(vec @ form.matrix.dot(vec))
    linear = float(form.linear @ vec)
    return quadratic + linear + form.constant


def combine_with_length_quadratic(
    A: np.ndarray | Any,
    linear_term: np.ndarray,
    medial_form: QuadraticForm,
    *,
    alpha: float,
) -> tuple[sparse.csr_matrix, np.ndarray, float]:
    """
    Combine Eq. (5) and Eq. (8) into Eq. (9): 0.5 * gamma^T H gamma + v^T gamma + c.
    """
    A_csr = sparse.csr_matrix(A, dtype=np.float64)
    l_arr = np.asarray(linear_term, dtype=np.float64).reshape(-1)

    A_shape = A_csr.shape
    if A_shape is None:
        raise ValueError("A shape is undefined")
    n_rows, n_cols = int(A_shape[0]), int(A_shape[1])
    if n_rows != n_cols:
        raise ValueError("A must be square")
    if n_rows != len(l_arr):
        raise ValueError("A and l must have compatible shapes")
    M_shape = medial_form.matrix.shape
    if M_shape is None:
        raise ValueError("medial_form.matrix shape is undefined")
    m_rows, m_cols = int(M_shape[0]), int(M_shape[1])
    if (m_rows, m_cols) != (n_rows, n_cols):
        raise ValueError("medial_form.matrix must match A shape")
    if medial_form.linear.shape != l_arr.shape:
        raise ValueError("medial_form.linear must match l shape")

    H = (2.0 * A_csr + alpha * medial_form.matrix).tocsr()
    v = l_arr + alpha * medial_form.linear
    c = alpha * medial_form.constant
    return H, v, c


def build_length_quadratic(
    curve: np.ndarray,
    *,
    length_weight: float = 1.0,
) -> tuple[Any, np.ndarray]:
    """
    Minimal closed-curve quadratic: E_L = w * sum_i ||gamma_{i+1} - gamma_i||^2.

    Returns Eq. (5)-style terms E_L = gamma^T A gamma + l^T gamma, with l = 0.
    """
    points = _as_points_3d("curve", curve)
    n = len(points)
    if n < 3:
        raise ValueError("curve must contain at least 3 points")
    if length_weight <= 0.0:
        raise ValueError("length_weight must be > 0")

    i_idx = np.arange(n)
    j_prev = (i_idx - 1) % n
    j_next = (i_idx + 1) % n

    rows = np.concatenate([i_idx, i_idx, i_idx])
    cols = np.concatenate([i_idx, j_prev, j_next])
    vals = np.concatenate(
        [
            np.full(n, 2.0 * length_weight, dtype=np.float64),
            np.full(n, -1.0 * length_weight, dtype=np.float64),
            np.full(n, -1.0 * length_weight, dtype=np.float64),
        ]
    )
    lap_1d = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    eye3 = sparse.eye(3, format="csr", dtype=np.float64)
    A = sparse.csr_matrix(sparse.kron(lap_1d, eye3, format="csr"))
    linear = np.zeros(3 * n, dtype=np.float64)
    return A, linear


def optimize_curve_newton(
    mesh: trimesh.Trimesh,
    initial_curve: np.ndarray,
    *,
    alpha: float = 1.0,
    length_weight: float = 1.0,
    n_iters: int = 5,
    step_size: float = 1.0,
    backtracking_coeff: float = 0.8,
    armijo_c1: float = 1e-4,
    max_line_search_steps: int = 20,
    damping: float = 1e-6,
    exclude_neighbor_hops: int = 1,
    tolerance: float = 1e-4,
    max_expand_steps: int = 16,
    max_binary_steps: int = 32,
    initial_radius_scale: float = 1.5,
) -> OptimizationResult:
    """
    Minimal paper-style outer loop: freeze implicit axis, solve quadratic step, repeat.
    """
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be in (0, 1]")
    if not 0.0 < backtracking_coeff < 1.0:
        raise ValueError("backtracking_coeff must be in (0, 1)")
    if not 0.0 < armijo_c1 < 1.0:
        raise ValueError("armijo_c1 must be in (0, 1)")
    if max_line_search_steps < 1:
        raise ValueError("max_line_search_steps must be >= 1")
    if damping <= 0.0:
        raise ValueError("damping must be > 0")

    curve = _as_points_3d("initial_curve", initial_curve).copy()
    energies: list[float] = []

    for _ in range(n_iters):
        medial = compute_medial_axis_energy(
            mesh,
            curve,
            exclude_neighbor_hops=exclude_neighbor_hops,
            tolerance=tolerance,
            max_expand_steps=max_expand_steps,
            max_binary_steps=max_binary_steps,
            initial_radius_scale=initial_radius_scale,
        )
        A, linear_len = build_length_quadratic(
            curve, length_weight=length_weight
        )
        H, v, c = combine_with_length_quadratic(
            A, linear_len, medial.quadratic, alpha=alpha
        )

        x = curve.reshape(-1)
        grad = H.dot(x) + v
        H_damped = H + damping * sparse.eye(H.shape[0], format="csr")
        delta = sparse_linalg.spsolve(H_damped, -grad)
        delta = np.asarray(delta, dtype=np.float64)

        current_energy = float(0.5 * x @ H.dot(x) + v @ x + c)
        grad_dot_delta = float(grad @ delta)

        t = step_size
        accepted_x = x.copy()
        accepted_energy = current_energy
        for _ in range(max_line_search_steps):
            candidate_x = x + t * delta
            candidate_curve = candidate_x.reshape(-1, 3)
            projected_curve = project_points_to_mesh(mesh, candidate_curve)
            projected_x = projected_curve.reshape(-1)
            candidate_energy = float(
                0.5 * projected_x @ H.dot(projected_x) + v @ projected_x + c
            )

            if (
                candidate_energy
                <= current_energy + armijo_c1 * t * grad_dot_delta
            ):
                accepted_x = projected_x
                accepted_energy = candidate_energy
                break
            t *= backtracking_coeff

        curve = accepted_x.reshape(-1, 3)
        energies.append(accepted_energy)

    return OptimizationResult(
        curve=curve, energies=np.asarray(energies, dtype=np.float64)
    )


def compute_medial_axis_energy(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
    *,
    exclude_neighbor_hops: int = 1,
    tolerance: float = 1e-4,
    max_expand_steps: int = 16,
    max_binary_steps: int = 32,
    initial_radius_scale: float = 1.5,
) -> MedialAxisEnergyResult:
    """
    Compute implicit medial axis points and return Eq. (7)/(8) quantities.
    """
    gamma = _as_points_3d("curve", curve)
    implicit_axis = compute_implicit_medial_axis(
        mesh,
        gamma,
        exclude_neighbor_hops=exclude_neighbor_hops,
        tolerance=tolerance,
        max_expand_steps=max_expand_steps,
        max_binary_steps=max_binary_steps,
        initial_radius_scale=initial_radius_scale,
    )

    weights = vertex_integration_weights(gamma)
    direct = medial_axis_energy(
        gamma,
        implicit_axis.m_plus,
        implicit_axis.m_minus,
        weights=weights,
    )
    quadratic = build_medial_axis_quadratic(
        gamma,
        implicit_axis.m_plus,
        implicit_axis.m_minus,
        weights=weights,
    )
    x = gamma.reshape(-1)
    energy = evaluate_quadratic_form(quadratic, x)

    return MedialAxisEnergyResult(
        energy=energy,
        direct_energy=direct,
        quadratic=quadratic,
        implicit_axis=implicit_axis,
        weights=weights,
    )
