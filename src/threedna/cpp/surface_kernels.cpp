#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <igl/point_mesh_squared_distance.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Vec3 = std::array<double, 3>;

namespace {

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 sub(const Vec3& a, const Vec3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 scale(const Vec3& a, double s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}

double dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double squared_norm(const Vec3& a) {
  return dot(a, a);
}

Vec3 barycentric_coordinates(
  const Vec3& p,
  const Vec3& a,
  const Vec3& b,
  const Vec3& c
) {
  const Vec3 v0 = sub(b, a);
  const Vec3 v1 = sub(c, a);
  const Vec3 v2 = sub(p, a);
  const double d00 = dot(v0, v0);
  const double d01 = dot(v0, v1);
  const double d11 = dot(v1, v1);
  const double d20 = dot(v2, v0);
  const double d21 = dot(v2, v1);
  const double denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) < 1e-20) {
    return {1.0, 0.0, 0.0};
  }
  const double v = (d11 * d20 - d01 * d21) / denom;
  const double w = (d00 * d21 - d01 * d20) / denom;
  const double u = 1.0 - v - w;
  return {u, v, w};
}

Vec3 closest_point_on_triangle(
  const Vec3& p,
  const Vec3& a,
  const Vec3& b,
  const Vec3& c
) {
  const Vec3 ab = sub(b, a);
  const Vec3 ac = sub(c, a);
  const Vec3 ap = sub(p, a);

  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return a;
  }

  const Vec3 bp = sub(p, b);
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return b;
  }

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return add(a, scale(ab, v));
  }

  const Vec3 cp = sub(p, c);
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return c;
  }

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return add(a, scale(ac, w));
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const Vec3 bc = sub(c, b);
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add(b, scale(bc, w));
  }

  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return add(a, add(scale(ab, v), scale(ac, w)));
}

py::array_t<double> project_points_to_mesh(
  py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
  py::array_t<long long, py::array::c_style | py::array::forcecast> faces,
  py::array_t<double, py::array::c_style | py::array::forcecast> points
) {
  if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
    throw std::invalid_argument("vertices must have shape (n_vertices, 3)");
  }
  if (faces.ndim() != 2 || faces.shape(1) != 3) {
    throw std::invalid_argument("faces must have shape (n_faces, 3)");
  }
  if (points.ndim() != 2 || points.shape(1) != 3) {
    throw std::invalid_argument("points must have shape (n_points, 3)");
  }

  const auto verts = vertices.unchecked<2>();
  const auto tris = faces.unchecked<2>();
  const auto query = points.unchecked<2>();

  const ssize_t n_vertices = vertices.shape(0);
  const ssize_t n_faces = faces.shape(0);
  const ssize_t n_points = points.shape(0);

  if (n_faces == 0) {
    throw std::invalid_argument("mesh has no faces");
  }

  auto out = py::array_t<double>({n_points, static_cast<ssize_t>(3)});
  auto projected = out.mutable_unchecked<2>();

  for (ssize_t i = 0; i < n_points; ++i) {
    const Vec3 p = {query(i, 0), query(i, 1), query(i, 2)};

    double best_dist2 = std::numeric_limits<double>::infinity();
    Vec3 best = p;

    for (ssize_t f = 0; f < n_faces; ++f) {
      const auto i0 = tris(f, 0);
      const auto i1 = tris(f, 1);
      const auto i2 = tris(f, 2);

      if (i0 < 0 || i0 >= n_vertices || i1 < 0 || i1 >= n_vertices || i2 < 0 || i2 >= n_vertices) {
        throw std::invalid_argument("faces contain out-of-range vertex indices");
      }

      const Vec3 a = {verts(i0, 0), verts(i0, 1), verts(i0, 2)};
      const Vec3 b = {verts(i1, 0), verts(i1, 1), verts(i1, 2)};
      const Vec3 c = {verts(i2, 0), verts(i2, 1), verts(i2, 2)};

      const Vec3 cp = closest_point_on_triangle(p, a, b, c);
      const double dist2 = squared_norm(sub(cp, p));
      if (dist2 < best_dist2) {
        best_dist2 = dist2;
        best = cp;
      }
    }

    projected(i, 0) = best[0];
    projected(i, 1) = best[1];
    projected(i, 2) = best[2];
  }

  return out;
}

py::tuple project_points_to_surface_points(
  const Eigen::Ref<const Eigen::MatrixXd>& vertices,
  const Eigen::Ref<const Eigen::MatrixXi>& faces,
  const Eigen::Ref<const Eigen::MatrixXd>& points
) {
  if (vertices.cols() != 3) {
    throw std::invalid_argument("vertices must have shape (n_vertices, 3)");
  }
  if (faces.cols() != 3) {
    throw std::invalid_argument("faces must have shape (n_faces, 3)");
  }
  if (points.cols() != 3) {
    throw std::invalid_argument("points must have shape (n_points, 3)");
  }
  if (faces.rows() == 0) {
    throw std::invalid_argument("mesh has no faces");
  }

  const int n_vertices = static_cast<int>(vertices.rows());
  for (int f = 0; f < faces.rows(); ++f) {
    for (int j = 0; j < 3; ++j) {
      const int idx = faces(f, j);
      if (idx < 0 || idx >= n_vertices) {
        throw std::invalid_argument("faces contain out-of-range vertex indices");
      }
    }
  }

  Eigen::VectorXd sqr_d;
  Eigen::VectorXi face_ids;
  Eigen::MatrixXd closest;
  igl::point_mesh_squared_distance(points, vertices, faces, sqr_d, face_ids, closest);

  Eigen::MatrixXd bary(points.rows(), 3);
  for (int i = 0; i < points.rows(); ++i) {
    const int f = face_ids(i);
    const Vec3 a = {vertices(faces(f, 0), 0), vertices(faces(f, 0), 1), vertices(faces(f, 0), 2)};
    const Vec3 b = {vertices(faces(f, 1), 0), vertices(faces(f, 1), 1), vertices(faces(f, 1), 2)};
    const Vec3 c = {vertices(faces(f, 2), 0), vertices(faces(f, 2), 1), vertices(faces(f, 2), 2)};
    const Vec3 p = {closest(i, 0), closest(i, 1), closest(i, 2)};
    const Vec3 bc = barycentric_coordinates(p, a, b, c);
    bary(i, 0) = bc[0];
    bary(i, 1) = bc[1];
    bary(i, 2) = bc[2];
  }

  return py::make_tuple(face_ids, bary, closest);
}

}  // namespace

PYBIND11_MODULE(_surface_kernels_cpp, m) {
  m.def(
    "project_points_to_mesh",
    &project_points_to_mesh,
    py::arg("vertices"),
    py::arg("faces"),
    py::arg("points")
  );
  m.def(
    "project_points_to_surface_points",
    &project_points_to_surface_points,
    py::arg("vertices"),
    py::arg("faces"),
    py::arg("points")
  );
}
