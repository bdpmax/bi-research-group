// ══════════════════════════════════════════════════════════════════
//  Athermal Quasistatic Sheared Voronoi Model — physics worker
//
//  JavaScript port of the MATLAB implementation by Junxiang Huang.
//
//  Primary references:
//    Huang, Cochran, Fielding, Marchetti, Bi,
//    "Shear-Driven Solidification and Nonlinear Elasticity in Epithelial
//     Tissues", Phys. Rev. Lett. 128, 178001 (2022).
//    Nguyen, Huang, Bi,
//    "Origin of yield stress and mechanical plasticity in model biological
//     tissues", Nat. Commun. 16, 3260 (2025).
//
//  Model: 2D Voronoi tissue, E = Σ [KA (A_i-A0)² + KP (P_i-P0)²].
//  Default KA=0 (perimeter-only); KP=1; A0=1; box L=√N (so A0=<A>).
//  Lees-Edwards periodic boundaries with strain γ.
//
//  AQS protocol: apply affine shear Δγ, then minimise energy with FIRE
//  to force tolerance. Stress σ_xy measured from ΔE/Δγ between relaxed
//  states. T1 events detected by tracking the Delaunay edge set.
// ══════════════════════════════════════════════════════════════════

importScripts('https://cdn.jsdelivr.net/npm/d3-delaunay@6/dist/d3-delaunay.min.js');
const { Delaunay } = self.d3;

// ── Parameters ────────────────────────────────────────────────────
let P = {
  N: 64,
  p0: 3.78,
  KA: 0.0,
  KP: 1.0,
  A0: 1.0,
  dGamma: 0.002,
  fireMax: 400,        // max FIRE iterations per strain step (interactive cap)
  fireTol: 5e-5,       // mean-force tolerance
  // FIRE parameters (from Matlab)
  dtMax: 0.10, dtMin: 0.001, alphaStart: 0.10, fAlpha: 0.99,
  finc: 1.10, fdec: 0.50, threshStep: 5, fireMass: 4.0,
};

// ── State ──────────────────────────────────────────────────────────
let rx, ry;
let gamma = 0;                    // accumulated strain (unbounded) — for display/plot
let gammaW = 0;                   // LEBC-wrapped strain in (-0.5, 0.5] — for physics
let L = 1;
let stepCount = 0;
let energy = 0, prevEnergy = 0, prevGamma = 0;
let cellArea, cellPerim;
let voroList = null;              // per-cell Voronoi polygon vertices (cell-local frame)
let edgeSet = new Set();          // current Delaunay edge set (for T1 detection)
let t1Events = [];                // recent T1s: {x,y,gamma,age}
let stressTrace = [];             // history: {gamma, sigma, energy}
const T1_MAX_HISTORY = 40;
const STRESS_MAX_POINTS = 5000;

function updateGammaW() {
  // Fold accumulated γ into (-0.5, 0.5] by subtracting nearest integer.
  // LEBC has periodic γ identification γ ≡ γ + 1, so this is physically
  // equivalent to the full accumulated strain (the lattice of sheared
  // images is the same under γ → γ + 1).
  gammaW = gamma - Math.round(gamma);
}

// Diagnostics from last FIRE call
let lastFireIters = 0;
let lastMeanForce = 0;

// ── Helpers ────────────────────────────────────────────────────────
function circumcircle(ax,ay,bx,by,cx,cy) {
  const D = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by));
  if (Math.abs(D) < 1e-14) return null;
  const ux = ((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/D;
  const uy = ((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/D;
  return { ux, uy };
}

// Lees-Edwards minimum-image displacement from (ax,ay) to (bx,by).
// Periodic box of side L. Uses the wrapped strain γ_w ∈ (-0.5, 0.5].
function minDispLEBC(ax, ay, bx, by) {
  let dx = bx - ax;
  let dy = by - ay;
  if (dy >  0.5 * L) { dy -= L; dx -= gammaW * L; }
  if (dy < -0.5 * L) { dy += L; dx += gammaW * L; }
  if (dx >  0.5 * L)   dx -= L;
  if (dx < -0.5 * L)   dx += L;
  return [dx, dy];
}

// Wrap a position back into [0,L)² respecting LEBC (γ_w).
function wrapLEBC(x, y) {
  if (y >= L) { y -= L; x -= gammaW * L; }
  else if (y < 0) { y += L; x += gammaW * L; }
  if (x >= L) x = ((x % L) + L) % L;
  else if (x < 0) x = ((x % L) + L) % L;
  if (y >= L || y < 0) y = ((y % L) + L) % L;
  return [x, y];
}

// ── Delaunay with 3×3 LEBC ghost tiling ────────────────────────────
function buildDelaunay() {
  const N = P.N;
  const flat = new Float64Array(9 * N * 2);
  const origs = new Int32Array(9 * N);
  let idx = 0;
  for (let di = -1; di <= 1; di++)
    for (let dj = -1; dj <= 1; dj++)
      for (let k = 0; k < N; k++) {
        // Top/bottom tiles carry an x-shift of ±γ_w L (wrapped strain)
        flat[idx*2]   = rx[k] + dj * L + di * gammaW * L;
        flat[idx*2+1] = ry[k] + di * L;
        origs[idx]    = k;
        idx++;
      }
  const centralStart = 4 * N;
  const d = new Delaunay(flat);
  const tris = d.triangles;
  const nT = tris.length / 3;
  const seen = new Set();
  const result = [];
  for (let t = 0; t < nT; t++) {
    const a = tris[t*3], b = tris[t*3+1], c = tris[t*3+2];
    const oa = origs[a], ob = origs[b], oc = origs[c];
    if (oa === ob || ob === oc || oa === oc) continue;
    const hasCentral =
      (a >= centralStart && a < centralStart + N) ||
      (b >= centralStart && b < centralStart + N) ||
      (c >= centralStart && c < centralStart + N);
    if (!hasCentral) continue;
    const key = [oa, ob, oc].sort((x,y) => x - y).join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    result.push([oa, ob, oc]);
  }
  return result;
}

function buildNeighboursAndVoro(tris) {
  const N = P.N;
  const triOfCell = Array.from({length:N}, () => []);
  for (const [a,b,c] of tris) {
    for (const [self, n1, n2] of [[a,b,c],[b,a,c],[c,a,b]]) {
      const sx = rx[self], sy = ry[self];
      const [d1x,d1y] = minDispLEBC(sx, sy, rx[n1], ry[n1]);
      const [d2x,d2y] = minDispLEBC(sx, sy, rx[n2], ry[n2]);
      const bx = sx+d1x, by = sy+d1y;
      const cx2 = sx+d2x, cy2 = sy+d2y;
      const cc = circumcircle(sx, sy, bx, by, cx2, cy2);
      if(!cc) continue;
      triOfCell[self].push({vx: cc.ux, vy: cc.uy, n1, n2});
    }
  }
  const nbrs = new Array(N);
  const voro = new Array(N);
  for (let i = 0; i < N; i++) {
    const ts = triOfCell[i];
    if (ts.length < 3) { nbrs[i]=[]; voro[i]=[]; continue; }
    const sx = rx[i], sy = ry[i];
    ts.sort((ta,tb) => Math.atan2(ta.vy-sy, ta.vx-sx) - Math.atan2(tb.vy-sy, tb.vx-sx));
    const nv = ts.length;
    const nOrder = new Array(nv);
    for (let m = 0; m < nv; m++) {
      const tm = ts[m];
      const tnext = ts[(m+1) % nv];
      const tmSet = new Set([tm.n1, tm.n2]);
      let shared = -1;
      for (const x of [tnext.n1, tnext.n2]) if (tmSet.has(x)) { shared = x; break; }
      nOrder[m] = shared;
    }
    nbrs[i] = nOrder;
    voro[i] = ts.map(t => ({x: t.vx, y: t.vy}));
  }
  return { nbrs, voro };
}

function computeGeometry(nbrs, voro) {
  const N = P.N;
  const areas = new Float64Array(N);
  const peris = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const vs = voro[i];
    const nv = vs.length;
    if (nv < 3) continue;
    let A = 0, Pe = 0;
    for (let m = 0; m < nv; m++) {
      const v0 = vs[m], v1 = vs[(m+1) % nv];
      A += v0.x * v1.y - v1.x * v0.y;
      const dx = v1.x - v0.x, dy = v1.y - v0.y;
      Pe += Math.sqrt(dx*dx + dy*dy);
    }
    areas[i] = Math.abs(A) * 0.5;
    peris[i] = Pe;
  }
  return { areas, peris };
}

// ── Force (SPV circumcenter-Jacobian chain rule, LEBC-aware) ───────
const FORCE_THRESH = 1e-12;
function computeForces(nbrsList, voroList, areas, peris, A0, P0) {
  const N = P.N;
  const fx = new Float64Array(N);
  const fy = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    const ns = nbrsList[i];
    const voro = voroList[i];
    const neigh = ns.length;
    if (neigh < 3) continue;

    const pix = rx[i], piy = ry[i];
    const dhdri = new Array(neigh);
    let [rijx, rijy] = minDispLEBC(pix, piy, rx[ns[neigh-1]], ry[ns[neigh-1]]);

    for (let nn = 0; nn < neigh; nn++) {
      const [rikx, riky] = minDispLEBC(pix, piy, rx[ns[nn]], ry[ns[nn]]);
      const rjkx = rikx - rijx, rjky = riky - rijy;
      const rij2 = rijx*rijx + rijy*rijy;
      const rik2 = rikx*rikx + riky*riky;
      const rijDrjk = rijx*rjkx + rijy*rjky;
      const rikDrjk = rikx*rjkx + riky*rjky;
      const betaD  = -rik2 * rijDrjk;
      const gammaD =  rij2 * rikDrjk;
      const cp = rijx*rjky - rijy*rjkx;
      const D  = 2*cp*cp;
      const zx = betaD*rijx + gammaD*rikx;
      const zy = betaD*rijy + gammaD*riky;
      const dbx =  2*rijDrjk*rikx + rik2*rjkx;
      const dby =  2*rijDrjk*riky + rik2*rjky;
      const dgx = -2*rikDrjk*rijx - rij2*rjkx;
      const dgy = -2*rikDrjk*rijy - rij2*rjky;
      const dDdri_over_D_x = -2*rjky/cp;
      const dDdri_over_D_y =  2*rjkx/cp;
      const bpg = betaD + gammaD;
      const iD = 1.0/D;
      const m00 = 1 + iD*(rijx*dbx + rikx*dgx - bpg - zx*dDdri_over_D_x);
      const m01 =     iD*(rijx*dby + rikx*dgy       - zx*dDdri_over_D_y);
      const m10 =     iD*(rijy*dbx + riky*dgx       - zy*dDdri_over_D_x);
      const m11 = 1 + iD*(rijy*dby + riky*dgy - bpg - zy*dDdri_over_D_y);
      dhdri[nn] = { m00, m01, m10, m11 };
      rijx = rikx; rijy = riky;
    }

    const Adiff_i = P.KA * (areas[i] - A0);
    const Pdiff_i = P.KP * (peris[i] - P0);
    let fsx = 0, fsy = 0;
    let vlast = voro[neigh-1];

    for (let nn = 0; nn < neigh; nn++) {
      const vcur  = voro[nn];
      const vnext = voro[(nn+1) % neigh];
      const baseNeigh  = ns[nn];
      const otherNeigh = ns[(nn-1+neigh) % neigh];

      const dAidvx = 0.5*(vlast.y - vnext.y);
      const dAidvy = 0.5*(vnext.x - vlast.x);
      let dlx = vlast.x - vcur.x, dly = vlast.y - vcur.y;
      let dnx = vcur.x - vnext.x, dny = vcur.y - vnext.y;
      const dlnorm = Math.max(Math.sqrt(dlx*dlx + dly*dly), FORCE_THRESH);
      const dnnorm = Math.max(Math.sqrt(dnx*dnx + dny*dny), FORCE_THRESH);
      const dPidvx = dlx/dlnorm - dnx/dnnorm;
      const dPidvy = dly/dlnorm - dny/dnnorm;

      const ns_k = nbrsList[baseNeigh];
      const nk = ns_k.length;
      let DT_other = -1, idx_in_k = -1;
      for (let n2 = 0; n2 < nk; n2++) {
        if (ns_k[n2] === otherNeigh) {
          idx_in_k = (n2+1) % nk;
          DT_other = ns_k[idx_in_k];
          break;
        }
      }
      if (DT_other < 0) continue;

      const vk_raw = voroList[baseNeigh][idx_in_k];
      // vcur and vk_raw live in DIFFERENT cell-local unwrapped frames.
      // Mirror SPV's trick: wrap both to the central box [0,L)² (under
      // LEBC for the y-coordinate), take the LEBC minimum-image
      // displacement, then place vk relative to vcur in cell i's frame.
      const wrapToBox = (x, y) => {
        if (y >= L) { y -= L; x -= gammaW * L; }
        else if (y < 0) { y += L; x += gammaW * L; }
        if (y < 0 || y >= L) y = ((y % L) + L) % L;
        if (x < 0 || x >= L) x = ((x % L) + L) % L;
        return [x, y];
      };
      const [vcurWx, vcurWy] = wrapToBox(vcur.x, vcur.y);
      const [vkWx, vkWy] = wrapToBox(vk_raw.x, vk_raw.y);
      const [dvox, dvoy] = minDispLEBC(vcurWx, vcurWy, vkWx, vkWy);
      const vox = vcur.x + dvox;
      const voy = vcur.y + dvoy;

      const dAkdvx = 0.5*(vnext.y - voy);
      const dAkdvy = 0.5*(vox - vnext.x);
      dlx = vnext.x - vcur.x; dly = vnext.y - vcur.y;
      dnx = vcur.x - vox;     dny = vcur.y - voy;
      const dlkn = Math.max(Math.sqrt(dlx*dlx + dly*dly), FORCE_THRESH);
      const dnkn = Math.max(Math.sqrt(dnx*dnx + dny*dny), FORCE_THRESH);
      const dPkdvx = dlx/dlkn - dnx/dnkn;
      const dPkdvy = dly/dlkn - dny/dnkn;

      const dAjdvx = 0.5*(voy - vlast.y);
      const dAjdvy = 0.5*(vlast.x - vox);
      dlx = vox - vcur.x;     dly = voy - vcur.y;
      dnx = vcur.x - vlast.x; dny = vcur.y - vlast.y;
      const dljn = Math.max(Math.sqrt(dlx*dlx + dly*dly), FORCE_THRESH);
      const dnjn = Math.max(Math.sqrt(dnx*dnx + dny*dny), FORCE_THRESH);
      const dPjdvx = dlx/dljn - dnx/dnjn;
      const dPjdvy = dly/dljn - dny/dnjn;

      const Adiff_k = P.KA * (areas[baseNeigh]  - A0);
      const Pdiff_k = P.KP * (peris[baseNeigh]  - P0);
      const Adiff_j = P.KA * (areas[otherNeigh] - A0);
      const Pdiff_j = P.KP * (peris[otherNeigh] - P0);

      const dEdvx = 2*Adiff_i*dAidvx + 2*Pdiff_i*dPidvx
                  + 2*Adiff_k*dAkdvx + 2*Pdiff_k*dPkdvx
                  + 2*Adiff_j*dAjdvx + 2*Pdiff_j*dPjdvx;
      const dEdvy = 2*Adiff_i*dAidvy + 2*Pdiff_i*dPidvy
                  + 2*Adiff_k*dAkdvy + 2*Pdiff_k*dPkdvy
                  + 2*Adiff_j*dAjdvy + 2*Pdiff_j*dPjdvy;

      const { m00, m01, m10, m11 } = dhdri[nn];
      fsx += dEdvx*m00 + dEdvy*m10;
      fsy += dEdvx*m01 + dEdvy*m11;
      vlast = vcur;
    }
    // cellGPU chain-rule convention: after the d3-delaunay polygon
    // orientation flip, fsx is the physical force (=-dE/dr_i), not the
    // gradient. Matches the SPV reference worker — do NOT add a minus.
    fx[i] = fsx;
    fy[i] = fsy;
  }
  return { fx, fy };
}

// ── Full geometry + forces + E for current (rx, ry, γ) ─────────────
function evaluate() {
  const tris = buildDelaunay();
  const { nbrs, voro } = buildNeighboursAndVoro(tris);
  const { areas, peris } = computeGeometry(nbrs, voro);
  // Targets: A0 = <A> = 1; P0 = p0 * sqrt(A0) = p0
  const A0 = P.A0;
  const P0 = P.p0 * Math.sqrt(A0);
  const { fx, fy } = computeForces(nbrs, voro, areas, peris, A0, P0);
  let E = 0;
  for (let i = 0; i < P.N; i++) {
    const dA = areas[i] - A0;
    const dP = peris[i] - P0;
    E += P.KA * dA * dA + P.KP * dP * dP;
  }
  return { tris, nbrs, voro, areas, peris, fx, fy, E };
}

// Build the edge set {min(i,j)*N+max(i,j)} from Delaunay triangles
function edgesFromTris(tris, N) {
  const s = new Set();
  for (const [a,b,c] of tris) {
    const pairs = [[a,b],[b,c],[a,c]];
    for (const [u,v] of pairs) {
      const lo = u < v ? u : v;
      const hi = u < v ? v : u;
      s.add(lo * N + hi);
    }
  }
  return s;
}

// ── FIRE minimisation ──────────────────────────────────────────────
function fireRelax(maxSteps) {
  const N = P.N;
  const vx = new Float64Array(N);
  const vy = new Float64Array(N);
  let dt = Math.sqrt(P.dtMax * P.dtMin);
  let alpha = P.alphaStart;
  let rightStep = 0;
  let meanF = 1e8;
  let iters = 0;
  let geom = null;

  for (iters = 0; iters < maxSteps; iters++) {
    geom = evaluate();
    const { fx, fy } = geom;

    // Mean force
    let mf = 0;
    for (let i = 0; i < N; i++) mf += Math.sqrt(fx[i]*fx[i] + fy[i]*fy[i]);
    meanF = mf / N;
    if (meanF < P.fireTol) break;

    // Power P = F·v
    let Pw = 0;
    for (let i = 0; i < N; i++) Pw += vx[i]*fx[i] + vy[i]*fy[i];

    // Normalized force direction
    const fhatx = new Float64Array(N), fhaty = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const fm = Math.sqrt(fx[i]*fx[i] + fy[i]*fy[i]);
      if (fm > 0) { fhatx[i] = fx[i]/fm; fhaty[i] = fy[i]/fm; }
    }

    if (Pw > 0 && rightStep > P.threshStep) {
      dt = Math.min(dt * P.finc, P.dtMax);
      alpha *= P.fAlpha;
      for (let i = 0; i < N; i++) {
        const vm = Math.sqrt(vx[i]*vx[i] + vy[i]*vy[i]);
        vx[i] = (1 - alpha) * vx[i] + alpha * vm * fhatx[i];
        vy[i] = (1 - alpha) * vy[i] + alpha * vm * fhaty[i];
      }
    } else if (Pw <= 0) {
      dt = Math.max(P.dtMin, dt * P.fdec);
      alpha = P.alphaStart;
      for (let i = 0; i < N; i++) { vx[i] = 0; vy[i] = 0; }
      rightStep = 0;
    }

    rightStep++;
    const invM = 1.0 / P.fireMass;
    for (let i = 0; i < N; i++) {
      vx[i] += fx[i] * invM * dt;
      vy[i] += fy[i] * invM * dt;
      rx[i] += vx[i] * dt;
      ry[i] += vy[i] * dt;
      // LEBC wrap
      [rx[i], ry[i]] = wrapLEBC(rx[i], ry[i]);
    }
  }

  lastFireIters = iters;
  lastMeanForce = meanF;
  return geom;
}

// ── Analytical shear stress (virial form) ─────────────────────────
// σ_xy = (1/A_box) Σ_cells Σ_{edges e in cell} T_e · (dx_e dy_e / |e|)
// with T_e = 2 K_P (P_c − P_0). Each shared edge is traversed by its
// two cells (once in each orientation: (dx)(dy) is invariant under
// inversion), so this sum automatically gives Σ_edges (T_c1 + T_c2)·…
function analyticalStress(voro, peris, P0) {
  const Abox = L * L;
  let sigma = 0;
  for (let i = 0; i < P.N; i++) {
    const vs = voro[i];
    const nv = vs ? vs.length : 0;
    if (nv < 3) continue;
    const T = 2 * P.KP * (peris[i] - P0);
    for (let k = 0; k < nv; k++) {
      const v0 = vs[k], v1 = vs[(k + 1) % nv];
      const dx = v1.x - v0.x, dy = v1.y - v0.y;
      const l = Math.sqrt(dx*dx + dy*dy);
      if (l > 1e-12) sigma += T * dx * dy / l;
    }
  }
  return sigma / Abox;
}

// ── AQS strain step ────────────────────────────────────────────────
function strainStep() {
  // 1) Apply affine shear. Shear only moves x; no y-boundary crossings,
  //    so plain x-wrap is sufficient.
  const dG = P.dGamma;
  for (let i = 0; i < P.N; i++) {
    rx[i] += dG * (ry[i] - 0.5 * L);
    if (rx[i] >= L) rx[i] -= L * Math.floor(rx[i] / L);
    else if (rx[i] < 0) rx[i] -= L * Math.floor(rx[i] / L);
  }
  gamma += dG;
  updateGammaW();

  // 2) FIRE relax (capped)
  const geom = fireRelax(P.fireMax);

  // 3) Stress from the virial/analytical formula at the relaxed state.
  //    (The energy-finite-difference form ΔE/(L²Δγ) is spuriously
  //    negative at T1s because E drops across the flip.)
  const A0 = P.A0;
  const P0 = P.p0 * Math.sqrt(A0);
  const sigma = geom ? analyticalStress(geom.voro, geom.peris, P0) : 0;
  const E = geom ? geom.E : energy;
  prevEnergy = E;
  prevGamma = gamma;
  energy = E;

  // 4) T1 detection: diff the Delaunay edge sets
  const newEdges = edgesFromTris(geom.tris, P.N);
  const lost = [], gained = [];
  if (edgeSet.size > 0) {
    for (const e of edgeSet) if (!newEdges.has(e)) lost.push(e);
    for (const e of newEdges) if (!edgeSet.has(e)) gained.push(e);
  }
  edgeSet = newEdges;

  // Record which cells participated (union of lost- and gained-edge
  // endpoints: for a standard T1 these are the 4 cells of the quartet).
  if (lost.length > 0 || gained.length > 0) {
    const cells = new Set();
    for (const e of lost)   { cells.add(Math.floor(e / P.N)); cells.add(e % P.N); }
    for (const e of gained) { cells.add(Math.floor(e / P.N)); cells.add(e % P.N); }
    t1Events.push({ cells: Array.from(cells), gamma, age: 0 });
  }
  for (const t of t1Events) t.age++;
  while (t1Events.length > T1_MAX_HISTORY) t1Events.shift();

  // 5) Stress trace
  stressTrace.push({ gamma, sigma, E: E / P.N });
  while (stressTrace.length > STRESS_MAX_POINTS) stressTrace.shift();

  stepCount++;
  return { geom, t1Count: lost.length, sigma };
}

// ── Init ───────────────────────────────────────────────────────────
function init() {
  const N = P.N;
  L = Math.sqrt(N);       // box side so ⟨A⟩ = 1 matches A0
  rx = new Float64Array(N); ry = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    rx[k] = Math.random() * L;
    ry[k] = Math.random() * L;
  }
  gamma = 0;
  gammaW = 0;
  stepCount = 0;
  stressTrace = [];
  t1Events = [];
  edgeSet = new Set();

  // Ground state relax at γ=0 (longer cap on the first relax)
  const geom = fireRelax(3 * P.fireMax);
  prevEnergy = geom ? geom.E : 0;
  energy = prevEnergy;
  prevGamma = 0;
  edgeSet = edgesFromTris(geom.tris, N);
  voroList = geom.voro;
  cellArea = geom.areas;
  cellPerim = geom.peris;
}

// ── Message handler ────────────────────────────────────────────────
function postState() {
  self.postMessage({
    type: 'state',
    rx: Array.from(rx),
    ry: Array.from(ry),
    voro: voroList,
    areas: Array.from(cellArea || []),
    perims: Array.from(cellPerim || []),
    gamma, gammaW, stepCount,
    L,
    energy, energyPerCell: energy / P.N,
    stressTrace: stressTrace.slice(),
    t1Events: t1Events.slice(),
    t1Count: t1Events.length,
    fireIters: lastFireIters,
    meanForce: lastMeanForce,
    p0: P.p0, KA: P.KA, KP: P.KP, dGamma: P.dGamma, N: P.N,
  });
}

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    if (m.params) Object.assign(P, m.params);
    init();
    postState();
  } else if (m.type === 'setParams') {
    Object.assign(P, m.params);
  } else if (m.type === 'step') {
    const n = m.nSteps || 1;
    let lastGeom = null;
    for (let s = 0; s < n; s++) {
      const r = strainStep();
      lastGeom = r.geom;
    }
    if (lastGeom) {
      voroList = lastGeom.voro;
      cellArea = lastGeom.areas;
      cellPerim = lastGeom.peris;
    }
    postState();
  }
};
