// voro3d.js
// Faithful JS port of the 3D periodic Voronoi cell model (Merkel & Manning 2018).
// Energy (dimensionless, Merkel & Manning 2018, Eq. 2):
//   e = sum_i [ kV (v_i - 1)^2 + kS (s_i - s0)^2 ],  v_i = N*V_i, s_i = N^(2/3)*S_i,
//   <V> = 1/N (unit periodic box).  ANALYTIC forces (gradient of e) + FIRE 2.0.
//
// Tessellation: replicate the N generators into their 27 periodic images, run a
// 3D Delaunay (delaunay-triangulate). Each Delaunay tet = a Voronoi vertex
// (its circumcenter); a Voronoi face between two cells is the ring of tets around
// a Delaunay edge. We compute each central cell's volume/surface and the exact
// gradient, accumulating force contributions onto cells (mod N) under PBC. This
// is the same physics as get_Cell_Force_full.m (FD-validated; see test_voro3d.mjs).
//
// The topology cache from the MATLAB "optimized" code is a pure speed trick
// (bit-identical physics); at demo sizes (N<100) we just rebuild each step.

// Tessellation comes from voro++ (WASM, walled container on image points).
import createVoroModule from './voro_wasm.js';
let MOD = null;
export async function initVoro() { if (!MOD) MOD = await createVoroModule(); return MOD; }

// ---------- tiny 3-vector helpers ----------
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const scal = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
const nrm = (a) => Math.sqrt(dot(a, a));

// ---------- seeded PRNG (mulberry32) ----------
export function makePRNG(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
export function makeConfig(N, seed) {
    const rng = makePRNG(seed);
    const cr = new Float64Array(3 * N);
    for (let k = 0; k < 3 * N; k++) cr[k] = rng();
    return cr;
}

// ---------- 3x3 inverse (rows m[0],m[1],m[2]) ----------
function inv3(m) {
    const a=m[0][0],b=m[0][1],c=m[0][2],
          d=m[1][0],e=m[1][1],f=m[1][2],
          g=m[2][0],h=m[2][1],i=m[2][2];
    const det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
    const id = 1/det;
    return [
        [(e*i-f*h)*id, (c*h-b*i)*id, (b*f-c*e)*id],
        [(f*g-d*i)*id, (a*i-c*g)*id, (c*d-a*f)*id],
        [(d*h-e*g)*id, (b*g-a*h)*id, (a*e-b*d)*id]
    ];
}

// circumcenter of tetra (p0,p1,p2,p3) and Jacobian B[k][b][e] = d c_b / d p_k,e
function circumJac(p0, p1, p2, p3) {
    const a = sub(p1,p0), b = sub(p2,p0), d = sub(p3,p0);
    const bxd = cross(b,d), dxa = cross(d,a), axb = cross(a,b);
    const denom = 2 * dot(a, bxd);
    const na2 = dot(a,a), nb2 = dot(b,b), nd2 = dot(d,d);
    const crel = [
        (na2*bxd[0] + nb2*dxa[0] + nd2*axb[0]) / denom,
        (na2*bxd[1] + nb2*dxa[1] + nd2*axb[1]) / denom,
        (na2*bxd[2] + nb2*dxa[2] + nd2*axb[2]) / denom
    ];
    const c = add(p0, crel);
    const Mi = inv3([a, b, d]);
    const cp = [sub(c,p0), sub(c,p1), sub(c,p2), sub(c,p3)];
    const rs = [Mi[0][0]+Mi[0][1]+Mi[0][2], Mi[1][0]+Mi[1][1]+Mi[1][2], Mi[2][0]+Mi[2][1]+Mi[2][2]];
    const B = [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]],
               [[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]];
    for (let bb=0; bb<3; bb++) for (let ee=0; ee<3; ee++) {
        B[0][bb][ee] =  rs[bb]      * cp[0][ee];
        B[1][bb][ee] = -Mi[bb][0]   * cp[1][ee];
        B[2][bb][ee] = -Mi[bb][1]   * cp[2][ee];
        B[3][bb][ee] = -Mi[bb][2]   * cp[3][ee];
    }
    return { c, B };
}

// order points (circumcenters) around an axis n, returning index permutation
function orderAroundAxis(P, n0) {
    const k = P.length;
    let g = [0,0,0];
    for (const p of P) { g[0]+=p[0]; g[1]+=p[1]; g[2]+=p[2]; }
    g = scal(g, 1/k);
    const n = scal(n0, 1/nrm(n0));
    let ref = Math.abs(n[0]) > 0.9 ? [0,1,0] : [1,0,0];
    let e1 = sub(ref, scal(n, dot(ref, n))); e1 = scal(e1, 1/nrm(e1));
    const e2 = cross(n, e1);
    const ang = P.map(p => { const q = sub(p, g); return Math.atan2(dot(q,e2), dot(q,e1)); });
    const idx = [...Array(k).keys()].sort((i,j) => ang[i]-ang[j]);
    return idx;
}

// ---------- energy + analytic force (+ render geometry) ----------
// returns {E, EperN, F (Float64Array 3N), Vc, Sc (per-cell), maxF, faces}
export function energyAndForce(cr, N, s0, kV, kS) {
    // 1) periodic images, CENTRAL first (ids 0..N-1) then the 26 neighbour images
    const pts = [], cellOf = [];
    const centralIdx = new Int32Array(N);
    for (let i=0; i<N; i++) { pts.push([cr[3*i],cr[3*i+1],cr[3*i+2]]); cellOf.push(i); centralIdx[i]=i; }
    for (let ox=-1; ox<=1; ox++) for (let oy=-1; oy<=1; oy++) for (let oz=-1; oz<=1; oz++) {
        if (ox===0 && oy===0 && oz===0) continue;
        for (let i=0; i<N; i++) { pts.push([cr[3*i]+ox, cr[3*i+1]+oy, cr[3*i+2]+oz]); cellOf.push(i); }
    }
    // 2) tessellation via voro++ (walled container on image points): each Voronoi
    //    vertex of a central cell -> its 4 generator image-point ids (Delaunay tets)
    const vpts = new MOD.VecDouble();
    for (const p of pts) { vpts.push_back(p[0]); vpts.push_back(p[1]); vpts.push_back(p[2]); }
    const tv = MOD.voroTets(vpts, N);
    const ntv = tv.size();
    const tets = [];
    for (let k=0; k<ntv; k+=4) tets.push([tv.get(k), tv.get(k+1), tv.get(k+2), tv.get(k+3)]);
    vpts.delete(); tv.delete();

    // 3) relevant tets (touch >=1 central point): circumcenter + Jacobian
    const centralSet = new Uint8Array(pts.length);
    for (let i=0; i<N; i++) centralSet[centralIdx[i]] = 1;

    const rt = [];                       // relevant tets: {v:[4], c, B}
    const inc = new Map();               // image point -> [relTet ids]
    for (const t of tets) {
        if (!(centralSet[t[0]] || centralSet[t[1]] || centralSet[t[2]] || centralSet[t[3]])) continue;
        const { c, B } = circumJac(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
        const id = rt.length;
        rt.push({ v: t, c, B });
        for (let a=0; a<4; a++) {
            const q = t[a];
            if (!inc.has(q)) inc.set(q, []);
            inc.get(q).push(id);
        }
    }

    // 4) per central cell: faces, V_i, S_i, energy, and gradient accumulation
    const grad = new Float64Array(3 * N);
    const Vc = new Float64Array(N), Sc = new Float64Array(N);
    const faces = [];
    const N23 = Math.pow(N, 2/3);

    for (let i=0; i<N; i++) {
        const ci = centralIdx[i];
        const tetsAt = inc.get(ci) || [];
        // group neighbor image point -> relTet ids sharing edge (ci, j)
        const nb = new Map();
        for (const rid of tetsAt) {
            for (const q of rt[rid].v) {
                if (q === ci) continue;
                if (!nb.has(q)) nb.set(q, []);
                nb.get(q).push(rid);
            }
        }
        const cell_faces = [];
        let Vi = 0, Si = 0;
        for (const [j, ringRaw] of nb) {
            if (ringRaw.length < 3) continue;                 // not a real face
            const Pv = ringRaw.map(rid => rt[rid].c);          // circumcenters
            const rij = sub(pts[j], pts[ci]);
            const ord = orderAroundAxis(Pv, rij);
            const ring = ord.map(k => ringRaw[k]);             // ordered relTet ids
            const Pord = ord.map(k => Pv[k]);
            const kk = ring.length;
            // vector area A = 1/2 sum v_m x v_{m+1}
            let A = [0,0,0];
            for (let m=0; m<kk; m++) {
                const cpv = cross(Pord[m], Pord[(m+1)%kk]);
                A[0]+=cpv[0]; A[1]+=cpv[1]; A[2]+=cpv[2];
            }
            A = scal(A, 0.5);
            const absA = nrm(A);
            if (absA < 1e-300) continue;
            const sgn = Math.sign(dot(rij, A)) || 1;
            const Aor = scal(A, sgn);
            const cf = (1/6) * sgn * dot(rij, A);              // pyramid volume (>=0)
            Si += absA; Vi += cf;
            cell_faces.push({ ring, Pord, A, absA, sgn, Aor, rij, j });
        }
        Vc[i] = Vi; Sc[i] = Si;
        const vi = N * Vi, si = N23 * Si;
        const WV = 2 * kV * N   * (vi - 1);                    // weight on dV_i/dr
        const WS = 2 * kS * N23 * (si - s0);                   // weight on dS_i/dr

        // gradient contributions from this cell's faces
        for (const f of cell_faces) {
            const { ring, Pord, A, absA, sgn, Aor, rij, j } = f;
            const Ahat = scal(A, 1/absA);
            const kk = ring.length;
            for (let m=0; m<kk; m++) {
                const vn = Pord[(m+1)%kk], vl = Pord[(m-1+kk)%kk];
                const w = sub(vn, vl);
                // dA_a/dv_m,b = 0.5 eps_{abg} w_g  (3x3 D[a][b])
                const D = [
                    [0,        0.5*w[2], -0.5*w[1]],
                    [-0.5*w[2], 0,        0.5*w[0]],
                    [ 0.5*w[1],-0.5*w[0], 0       ]
                ];
                // per-vertex 3-vec gradient of (WS*|A| + WV*cf) wrt v_m
                const gv = [0,0,0];
                for (let b=0; b<3; b++) {
                    let surf=0, vol=0;
                    for (let a=0; a<3; a++) { surf += Ahat[a]*D[a][b]; vol += rij[a]*D[a][b]; }
                    gv[b] = WS*surf + WV*(1/6)*sgn*vol;
                }
                // chain onto the tet's image points: dE/dp = gv . (dv_m/dp = B)
                const tet = rt[ring[m]];
                for (let a=0; a<4; a++) {
                    const cl = cellOf[tet.v[a]];
                    const Bb = tet.B[a];
                    for (let e=0; e<3; e++)
                        grad[3*cl+e] += gv[0]*Bb[0][e] + gv[1]*Bb[1][e] + gv[2]*Bb[2][e];
                }
            }
            // explicit volume P1: d cf/d pos(ci) = -(1/6)Aor ; d/d pos(j)=+(1/6)Aor
            const cj = cellOf[j];
            for (let e=0; e<3; e++) {
                grad[3*i+e]  += WV * (-(1/6)*Aor[e]);
                grad[3*cj+e] += WV * ( (1/6)*Aor[e]);
            }
        }

        // render geometry: ordered face vertex coords
        for (const f of cell_faces) faces.push({ cell: i, verts: f.Pord });
    }

    // energy + force
    let E = 0;
    for (let i=0; i<N; i++) E += kV*(N*Vc[i]-1)**2 + kS*(N23*Sc[i]-s0)**2;
    const F = new Float64Array(3*N);
    for (let k=0; k<3*N; k++) F[k] = -grad[k];
    let maxF = 0;
    for (let i=0; i<N; i++) {
        const fm = Math.hypot(F[3*i], F[3*i+1], F[3*i+2]);
        if (fm > maxF) maxF = fm;
    }
    return { E, EperN: E/N, F, Vc, Sc, maxF, faces };
}

// ---------- FIRE 2.0 (with finite guard), returns relaxed state + history ----------
export function fireStep(state) {
    // single FIRE step on state {cr,N,s0,kV,kS,v,dt,alpha,npos,it,params}
    const p = state.params;
    const r = energyAndForce(state.cr, state.N, state.s0, state.kV, state.kS);
    state.it++;
    state.last = r;
    if (!isFinite(r.maxF)) { state.status = 3; state.done = true; return r; }
    if (r.maxF <= p.ftol) { state.status = 1; state.done = true; return r; }
    if (state.it >= p.maxSteps) { state.status = 2; state.done = true; return r; }

    const F = r.F, v = state.v, n = 3*state.N;
    let P = 0; for (let k=0;k<n;k++) P += F[k]*v[k];
    if (P > 0) {
        state.npos++;
        if (state.npos > p.nDelay) { state.dt = Math.min(state.dt*p.fInc, p.dtMax); state.alpha *= p.fAlpha; }
    } else {
        state.npos = 0;
        if (!(p.initialDelay && state.it <= p.nDelay)) { state.dt = Math.max(state.dt*p.fDec, p.dtMin); state.alpha = p.alpha0; }
        for (let k=0;k<n;k++) state.cr[k] -= 0.5*state.dt*v[k];
        v.fill(0);
    }
    // semi-implicit Euler + velocity mixing
    let fn=0, vn=0;
    for (let k=0;k<n;k++){ v[k]+=state.dt*F[k]; fn+=F[k]*F[k]; }
    fn = Math.sqrt(fn);
    for (let k=0;k<n;k++) vn += v[k]*v[k];
    vn = Math.sqrt(vn);
    if (fn>0) { const a=state.alpha; for (let k=0;k<n;k++) v[k] = (1-a)*v[k] + a*vn*F[k]/fn; }
    for (let k=0;k<n;k++){ let x = state.cr[k] + state.dt*v[k]; x -= Math.floor(x); state.cr[k] = x; } // PBC wrap
    return r;
}

export function makeState(N, s0, kV, kS, seed, params) {
    const def = { ftol:1e-6, maxSteps:2000, dtInit:0.02, dtMax:0.20, dtMin:1e-5,
                  nDelay:20, fInc:1.1, fDec:0.5, alpha0:0.10, fAlpha:0.99, initialDelay:true };
    const p = Object.assign(def, params||{});
    return { cr: makeConfig(N, seed), N, s0, kV, kS, v: new Float64Array(3*N),
             dt: p.dtInit, alpha: p.alpha0, npos: 0, it: 0, status: 0, done: false, params: p };
}

export function relax(N, s0, kV, kS, seed, params) {
    const st = makeState(N, s0, kV, kS, seed, params);
    while (!st.done) fireStep(st);
    const r = st.last;
    return { cr: st.cr, EperN: r.EperN, maxF: r.maxF, status: st.status, nevals: st.it,
             Vc: r.Vc, Sc: r.Sc };
}
