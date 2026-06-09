// voro_wrap.cpp — voro++ (Emscripten) on a set of IMAGE points (walled container).
// PBC is handled by image replication on the JS side (first N points are the
// central cells). For every central cell we read its Voronoi vertices and emit
// each vertex's 4 generator IMAGE-point ids (cell + its 3 face-neighbours at that
// vertex), de-duplicated by sorted quadruple. These are exactly the Delaunay
// tetrahedra incident to the central cells -> fed to the validated, image-based
// energyAndForce in voro3d.js. Using image-point ids (not periodic cell ids)
// keeps multi-image neighbours unambiguous.
#include "voro++.hh"
#include <emscripten/bind.h>
#include <vector>
#include <set>
#include <array>
#include <algorithm>
#include <cmath>
using namespace voro;
using namespace emscripten;

std::vector<int> voroTets(const std::vector<double>& pts, int N) {
    int M = (int)pts.size() / 3;
    double lo[3] = {1e30,1e30,1e30}, hi[3] = {-1e30,-1e30,-1e30};
    for (int k = 0; k < M; k++) for (int d = 0; d < 3; d++) {
        double v = pts[3*k+d]; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v;
    }
    for (int d = 0; d < 3; d++) { lo[d] -= 0.1; hi[d] += 0.1; }
    int nb = std::max(1, (int)std::round(std::cbrt((double)M / 5.0)));
    container con(lo[0],hi[0], lo[1],hi[1], lo[2],hi[2], nb,nb,nb, false,false,false, 8);
    for (int k = 0; k < M; k++) con.put(k, pts[3*k], pts[3*k+1], pts[3*k+2]);

    std::set<std::array<int,4>> seen;
    std::vector<int> out;
    voronoicell_neighbor c;
    c_loop_all vl(con);
    if (vl.start()) do {
        int id = vl.pid();
        if (id >= N) continue;                 // central cells only (first N)
        if (!con.compute_cell(c, vl)) continue;
        std::vector<int> nbr, fv;
        c.neighbors(nbr);
        c.face_vertices(fv);
        int nv = c.p;
        std::vector<std::vector<int>> vn(nv);
        int fp = 0, f = 0;
        while (fp < (int)fv.size()) {
            int k = fv[fp++];
            for (int a = 0; a < k; a++) vn[fv[fp+a]].push_back(nbr[f]);
            fp += k; f++;
        }
        for (int v = 0; v < nv; v++) {
            if ((int)vn[v].size() != 3) continue;
            if (vn[v][0] < 0 || vn[v][1] < 0 || vn[v][2] < 0) continue;  // wall-touching
            std::array<int,4> q = {id, vn[v][0], vn[v][1], vn[v][2]};
            std::sort(q.begin(), q.end());
            if (seen.insert(q).second) {
                out.push_back(q[0]); out.push_back(q[1]); out.push_back(q[2]); out.push_back(q[3]);
            }
        }
    } while (vl.inc());
    return out;
}

EMSCRIPTEN_BINDINGS(voro_mod) {
    register_vector<double>("VecDouble");
    register_vector<int>("VecInt");
    function("voroTets", &voroTets);
}
