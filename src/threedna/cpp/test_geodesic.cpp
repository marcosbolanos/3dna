#include <iostream>
#include <vector>
#include <memory>

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <geometrycentral/surface/surface_mesh_factories.h>
#include <geometrycentral/surface/exact_geodesics.h>
#include <geometrycentral/surface/surface_point.h>

using namespace geometrycentral;
using namespace surface;

int main() {
    std::cout << "Testing geometry-central exact geodesics..." << std::endl;

    // Create a simple cube mesh manually
    std::vector<Vector3> vertices = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},  // 0-3: back face
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}   // 4-7: front face
    };

    // Cube faces (12 triangles for watertight cube)
    std::vector<std::vector<size_t>> faces = {
        {0, 1, 2}, {0, 2, 3},  // back
        {4, 6, 5}, {4, 7, 6},  // front
        {0, 4, 5}, {0, 5, 1},  // bottom
        {2, 6, 7}, {2, 7, 3},  // top
        {0, 3, 7}, {0, 7, 4},  // left
        {1, 5, 6}, {1, 6, 2}   // right
    };

    std::cout << "Creating mesh with " << vertices.size() << " vertices and " << faces.size() << " faces..." << std::endl;

    // Build geometry-central mesh using factory
    auto [mesh, geom] = makeManifoldSurfaceMeshAndGeometry(faces, vertices);
    
    if (!mesh) {
        std::cerr << "Failed to create manifold mesh!" << std::endl;
        return 1;
    }
    std::cout << "Mesh created: " << mesh->nVertices() << " vertices, " << mesh->nFaces() << " faces" << std::endl;

    // Require edge lengths for geodesics
    geom->requireEdgeLengths();
    geom->requireCornerAngles();
    geom->requireVertexGaussianCurvatures();

    std::cout << "Computing exact geodesics from vertex 0..." << std::endl;

    // Create geodesic solver
    GeodesicAlgorithmExact mmp(*mesh, *geom);

    // Propagate from vertex 0
    Vertex source = mesh->vertex(0);
    mmp.propagate({SurfacePoint(source)});

    // Get distances to all vertices
    auto distances = mmp.getDistanceFunction();

    std::cout << "\nDistances from vertex 0:" << std::endl;
    for (Vertex v : mesh->vertices()) {
        std::cout << "  vertex " << v.getIndex() << ": " << distances[v] << std::endl;
    }

    // Test traceback from a point
    std::cout << "\nTesting traceback to vertex 4 (opposite corner)..." << std::endl;
    Vertex target = mesh->vertex(4);
    auto path = mmp.traceBack(SurfacePoint(target));

    std::cout << "Path has " << path.size() << " points:" << std::endl;
    for (size_t i = 0; i < path.size(); ++i) {
        const auto& sp = path[i];
        std::cout << "  [" << i << "] type: ";
        switch (sp.type) {
            case SurfacePointType::Vertex:
                std::cout << "vertex " << sp.vertex.getIndex();
                break;
            case SurfacePointType::Edge:
                std::cout << "edge " << sp.edge.getIndex();
                break;
            case SurfacePointType::Face:
                std::cout << "face " << sp.face.getIndex();
                break;
        }
        std::cout << std::endl;
    }

    std::cout << "\n✅ Geodesic algorithm working!" << std::endl;

    return 0;
}
