#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// IMPORTANT: Include pybind11/eigen.h BEFORE any geometry-central headers
// to get proper Eigen->Python conversions
#include <pybind11/eigen.h>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <geometrycentral/surface/surface_mesh_factories.h>
#include <geometrycentral/surface/exact_geodesics.h>
#include <geometrycentral/surface/surface_point.h>

#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace geometrycentral;
using namespace surface;

struct SurfacePointData {
    int type = 0;
    int index = 0;
    Eigen::Vector3d coords = Eigen::Vector3d::Zero();
};


class GeodesicMesh {
public:
    GeodesicMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
        // Use Eigen matrices directly with template function
        auto [mesh_ptr, geom_ptr] = makeManifoldSurfaceMeshAndGeometry(V, F);
        mesh = std::move(mesh_ptr);
        geom = std::move(geom_ptr);

        // Require geometry for geodesics
        geom->requireEdgeLengths();
        geom->requireCornerAngles();
        geom->requireVertexGaussianCurvatures();

        // Create solver
        solver = std::make_unique<GeodesicAlgorithmExact>(*mesh, *geom);
    }

    void propagate_from_vertices(const std::vector<int>& vertex_indices) {
        std::vector<SurfacePoint> sources;
        for (int idx : vertex_indices) {
            sources.push_back(SurfacePoint(mesh->vertex(idx)));
        }
        solver->propagate(sources);
    }

    Eigen::VectorXd distance_from_vertices(const std::vector<int>& vertex_indices) {
        propagate_from_vertices(vertex_indices);
        auto distances = solver->getDistanceFunction();
        
        Eigen::VectorXd result(mesh->nVertices());
        for (Vertex v : mesh->vertices()) {
            result[v.getIndex()] = distances[v];
        }
        return result;
    }

    SurfacePointData make_vertex_point(int vertex_idx) const {
        validate_vertex_index(vertex_idx);
        SurfacePointData out;
        out.type = 0;
        out.index = vertex_idx;
        return out;
    }

    SurfacePointData make_edge_point(int edge_idx, double t_edge) const {
        validate_edge_index(edge_idx);
        if (t_edge < 0.0 || t_edge > 1.0) {
            throw std::invalid_argument("edge parameter t must be in [0, 1]");
        }
        SurfacePointData out;
        out.type = 1;
        out.index = edge_idx;
        out.coords = Eigen::Vector3d(t_edge, 0.0, 0.0);
        return out;
    }

    SurfacePointData make_face_point(int face_idx, double b0, double b1, double b2) const {
        validate_face_index(face_idx);
        const double sum = b0 + b1 + b2;
        if (std::abs(sum - 1.0) > 1e-6) {
            throw std::invalid_argument("face barycentric coordinates must sum to 1");
        }
        SurfacePointData out;
        out.type = 2;
        out.index = face_idx;
        out.coords = Eigen::Vector3d(b0, b1, b2);
        return out;
    }

    std::vector<SurfacePointData> trace_path_points(
        const SurfacePointData& source,
        const SurfacePointData& target
    ) {
        const SurfacePoint source_sp = to_surface_point(source);
        const SurfacePoint target_sp = to_surface_point(target);
        solver->propagate({source_sp});
        const auto path = solver->traceBack(target_sp);

        std::vector<SurfacePointData> result;
        result.reserve(path.size());
        for (const auto& sp : path) {
            result.push_back(from_surface_point(sp));
        }
        return result;
    }

    std::vector<std::tuple<int, int, Eigen::Vector3d>> trace_path(int source_idx, int target_idx) {
        const SurfacePointData source = make_vertex_point(source_idx);
        const SurfacePointData target = make_vertex_point(target_idx);
        const auto path = trace_path_points(source, target);

        std::vector<std::tuple<int, int, Eigen::Vector3d>> result;
        result.reserve(path.size());
        for (const auto& sp : path) {
            if (sp.type == 1) {
                result.emplace_back(sp.type, sp.index, Eigen::Vector3d(1.0 - sp.coords[0], sp.coords[0], 0.0));
            } else {
                result.emplace_back(sp.type, sp.index, sp.coords);
            }
        }
        return result;
    }

    // Get vertex positions as numpy array
    Eigen::MatrixXd vertex_positions() const {
        Eigen::MatrixXd result(mesh->nVertices(), 3);
        for (Vertex v : mesh->vertices()) {
            geometrycentral::Vector3 pos = geom->vertexPositions[v];
            result(v.getIndex(), 0) = pos.x;
            result(v.getIndex(), 1) = pos.y;
            result(v.getIndex(), 2) = pos.z;
        }
        return result;
    }

    int n_vertices() const { return mesh->nVertices(); }
    int n_faces() const { return mesh->nFaces(); }
    int n_edges() const { return mesh->nEdges(); }

    Eigen::MatrixXi face_vertex_indices() const {
        Eigen::MatrixXi result(mesh->nFaces(), 3);
        int i = 0;
        for (Face f : mesh->faces()) {
            int j = 0;
            for (Vertex v : f.adjacentVertices()) {
                result(i, j++) = v.getIndex();
            }
            i++;
        }
        return result;
    }

    Eigen::MatrixXi edge_vertex_indices() const {
        Eigen::MatrixXi result(mesh->nEdges(), 2);
        int i = 0;
        for (Edge e : mesh->edges()) {
            result(i, 0) = e.firstVertex().getIndex();
            result(i, 1) = e.secondVertex().getIndex();
            i++;
        }
        return result;
    }

private:
    void validate_vertex_index(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(mesh->nVertices())) {
            throw std::out_of_range("vertex index out of range");
        }
    }

    void validate_edge_index(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(mesh->nEdges())) {
            throw std::out_of_range("edge index out of range");
        }
    }

    void validate_face_index(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(mesh->nFaces())) {
            throw std::out_of_range("face index out of range");
        }
    }

    SurfacePoint to_surface_point(const SurfacePointData& sp) const {
        switch (sp.type) {
            case 0:
                validate_vertex_index(sp.index);
                return SurfacePoint(mesh->vertex(static_cast<size_t>(sp.index)));
            case 1:
                validate_edge_index(sp.index);
                if (sp.coords[0] < 0.0 || sp.coords[0] > 1.0) {
                    throw std::invalid_argument("edge parameter t must be in [0, 1]");
                }
                return SurfacePoint(mesh->edge(static_cast<size_t>(sp.index)), sp.coords[0]);
            case 2:
                validate_face_index(sp.index);
                return SurfacePoint(
                    mesh->face(static_cast<size_t>(sp.index)),
                    geometrycentral::Vector3{sp.coords[0], sp.coords[1], sp.coords[2]}
                );
            default:
                throw std::invalid_argument("surface point type must be 0 (vertex), 1 (edge), or 2 (face)");
        }
    }

    SurfacePointData from_surface_point(const SurfacePoint& sp) const {
        SurfacePointData out;
        switch (sp.type) {
            case SurfacePointType::Vertex:
                out.type = 0;
                out.index = static_cast<int>(sp.vertex.getIndex());
                out.coords = Eigen::Vector3d::Zero();
                break;
            case SurfacePointType::Edge:
                out.type = 1;
                out.index = static_cast<int>(sp.edge.getIndex());
                out.coords = Eigen::Vector3d(sp.tEdge, 0.0, 0.0);
                break;
            case SurfacePointType::Face:
                out.type = 2;
                out.index = static_cast<int>(sp.face.getIndex());
                out.coords = Eigen::Vector3d(sp.faceCoords.x, sp.faceCoords.y, sp.faceCoords.z);
                break;
        }
        return out;
    }

    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::unique_ptr<GeodesicAlgorithmExact> solver;
};


PYBIND11_MODULE(_geodesic_cpp, m) {
    m.doc() = "Exact geodesic computations using geometry-central";

    py::class_<SurfacePointData>(m, "SurfacePoint")
        .def(py::init<>())
        .def_readwrite("type", &SurfacePointData::type)
        .def_readwrite("index", &SurfacePointData::index)
        .def_readwrite("coords", &SurfacePointData::coords);

    py::class_<GeodesicMesh>(m, "GeodesicMesh")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXi&>(),
             "Create geodesic mesh from vertices and faces",
             py::arg("V"), py::arg("F"))
        .def("n_vertices", &GeodesicMesh::n_vertices)
        .def("n_faces", &GeodesicMesh::n_faces)
        .def("n_edges", &GeodesicMesh::n_edges)
        .def("vertex_positions", &GeodesicMesh::vertex_positions)
        .def("face_vertex_indices", &GeodesicMesh::face_vertex_indices)
        .def("edge_vertex_indices", &GeodesicMesh::edge_vertex_indices)
        .def("make_vertex_point", &GeodesicMesh::make_vertex_point, py::arg("vertex"))
        .def("make_edge_point", &GeodesicMesh::make_edge_point, py::arg("edge"), py::arg("t"))
        .def("make_face_point", &GeodesicMesh::make_face_point,
             py::arg("face"), py::arg("b0"), py::arg("b1"), py::arg("b2"))
        .def("distance_from_vertices", &GeodesicMesh::distance_from_vertices,
             "Compute geodesic distances from source vertices to all vertices",
             py::arg("sources"))
        .def("trace_path_points", &GeodesicMesh::trace_path_points,
             "Trace exact geodesic path between arbitrary surface points",
             py::arg("source"), py::arg("target"))
        .def("trace_path", &GeodesicMesh::trace_path,
             "Trace exact geodesic path from source to target vertex",
             py::arg("source"), py::arg("target"));
}
