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

namespace py = pybind11;
using namespace geometrycentral;
using namespace surface;

// Use Eigen::Map to avoid Vector3 conflicts
// We'll convert to geometry-central types internally


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

    // Trace exact geodesic path from source vertex to target vertex
    // Returns path as list of (type, index, barycentric) tuples
    // type: 0=vertex, 1=edge, 2=face
    std::vector<std::tuple<int, int, Eigen::Vector3d>> trace_path(int source_idx, int target_idx) {
        // First propagate from source
        solver->propagate({SurfacePoint(mesh->vertex(source_idx))});
        
        // Then trace back to target
        auto path = solver->traceBack(SurfacePoint(mesh->vertex(target_idx)));
        
        std::vector<std::tuple<int, int, Eigen::Vector3d>> result;
        result.reserve(path.size());
        
        for (const auto& sp : path) {
            Eigen::Vector3d bary(0, 0, 0);
            int type, idx;
            
            switch (sp.type) {
                case SurfacePointType::Vertex:
                    type = 0;
                    idx = sp.vertex.getIndex();
                    bary = Eigen::Vector3d(1, 0, 0);
                    break;
                case SurfacePointType::Edge:
                    type = 1;
                    idx = sp.edge.getIndex();
                    bary = Eigen::Vector3d(1 - sp.tEdge, sp.tEdge, 0);
                    break;
                case SurfacePointType::Face:
                    type = 2;
                    idx = sp.face.getIndex();
                    bary = Eigen::Vector3d(sp.faceCoords.x, sp.faceCoords.y, sp.faceCoords.z);
                    break;
            }
            result.emplace_back(type, idx, bary);
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
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
    std::unique_ptr<GeodesicAlgorithmExact> solver;
};


PYBIND11_MODULE(_geodesic_cpp, m) {
    m.doc() = "Exact geodesic computations using geometry-central";

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
        .def("distance_from_vertices", &GeodesicMesh::distance_from_vertices,
             "Compute geodesic distances from source vertices to all vertices",
             py::arg("sources"))
        .def("trace_path", &GeodesicMesh::trace_path,
             "Trace exact geodesic path from source to target vertex",
             py::arg("source"), py::arg("target"));
}
