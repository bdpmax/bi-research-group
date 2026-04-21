// ══════════════════════════════════════════════════════════════
//  Active Finite Voronoi (AFV) physics worker
//
//  JavaScript port of pyafv (https://github.com/wwang721/pyafv):
//    [1] W. Wang and B. A. Camley,
//        Divergence of detachment forces in the finite Voronoi model,
//        arXiv preprint arXiv:2604.15481 (2026).
//
//  Underlying model:
//    [2] Huang, Levine, Bi — Soft Matter 19, 9389 (2023)
//    [3] Teomy, Kessler, Levine — Phys. Rev. E 98, 042418 (2018)
//
//  Geometry: standard Voronoi tessellation of cell centres, intersected
//  with a disk of radius ℓ around each cell. Boundary consists of
//  straight edges (shared with neighbours inside range ℓ) and circular
//  arcs (where the Voronoi region extends beyond ℓ; "cell–medium" edges).
//
//  Energy:  E = Σᵢ [ K_A (A_i − A_0)² + K_P (P_i − P_0)² ] + Λ Σᵢ L_i^arc
//
//  Dynamics:  ṙᵢ = μ F_i + v_0 n̂_i,   θ̇_i = η,  ⟨η(t)η(t')⟩ = 2 D_r δ(t-t')
//
//  Forces: computed numerically by central finite differences of the
//  total energy (small ε, Voronoi rebuilt per perturbation via
//  d3-delaunay). This avoids porting the full analytical pyafv
//  derivative machinery; accuracy is controlled by ε.
// ══════════════════════════════════════════════════════════════

importScripts('https://cdn.jsdelivr.net/npm/d3-delaunay@6/dist/d3-delaunay.min.js');
const { Delaunay } = self.d3;

// ── Parameters ────────────────────────────────────────────────
let P = {
  N:        36,
  r:        1.00,    // maximal interaction radius ℓ
  A0:       Math.PI, // preferred area
  P0:       4.80,    // preferred perimeter (p0 = 4.80/√π ≈ 2.71)
  KA:       1.0,
  KP:       1.0,
  Lambda:   0.2,     // tension on non-contacting (arc) edges
  v0:       0.05,
  Dr:       0.10,
  mu:       1.0,
  dt:       0.02,
  spacing:  1.80,    // initial hex-lattice spacing
  boxPad:   3.0,     // padding around initial lattice
};

// ── State ──────────────────────────────────────────────────────
let rx, ry, theta, spd;
let cellArea, cellPerim, cellArcLen;
let cellPieces;       // per-cell boundary pieces for rendering
let dispX, dispY;     // unwrapped displacement (not used since no PBC, but useful for MSD)
let stepCount = 0;
let boxW = 1, boxH = 1;

// ── Helpers ────────────────────────────────────────────────────
let _g = null;
function gauss() {
  if (_g !== null) { const v = _g; _g = null; return v; }
  let u, v, s;
  do { u = Math.random()*2-1; v = Math.random()*2-1; s = u*u+v*v; } while (s>=1 || s===0);
  const f = Math.sqrt(-2*Math.log(s)/s); _g = v*f; return u*f;
}

function lineCircleIntersect(a, b, cx, cy, r) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const ex = a[0] - cx,   ey = a[1] - cy;
  const A = dx*dx + dy*dy;
  const B = 2 * (ex*dx + ey*dy);
  const C = ex*ex + ey*ey - r*r;
  const disc = B*B - 4*A*C;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t1 = (-B - sq) / (2*A);
  const t2 = (-B + sq) / (2*A);
  const t = (t1 >= -1e-9 && t1 <= 1 + 1e-9) ? t1 : t2;
  return [a[0] + t*dx, a[1] + t*dy];
}

function arcAngleCCW(a, b, cx, cy) {
  const ta = Math.atan2(a[1]-cy, a[0]-cx);
  const tb = Math.atan2(b[1]-cy, b[0]-cx);
  let da = tb - ta;
  while (da < 0)        da += 2*Math.PI;
  while (da >= 2*Math.PI) da -= 2*Math.PI;
  return da;
}

// ══════════════════════════════════════════════════════════════
//  Finite Voronoi geometry: clip the standard Voronoi polygon
//  of cell i by a disk of radius ℓ around the cell centre.
//
//  Correct general algorithm: for each polygon edge, compute its
//  in-disk portion (zero or one contiguous segment — disks are
//  convex, so a line crosses at most twice). Stitch the resulting
//  "inside segments" with circular arcs where consecutive segments
//  leave and re-enter the disk.
// ══════════════════════════════════════════════════════════════
function pointInPolygon(px, py, verts) {
  let inside = false;
  const n = verts.length;
  for (let i = 0, j = n-1; i < n; j = i++) {
    const xi = verts[i][0], yi = verts[i][1];
    const xj = verts[j][0], yj = verts[j][1];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function clipAndMeasure(poly, cx, cy, r) {
  if (!poly || poly.length < 4) {
    const A = Math.PI * r * r, Pcirc = 2 * Math.PI * r;
    return { area: A, perim: Pcirc, arcLen: Pcirc, pieces: [{ type: 'fulldisk', cx, cy, r }] };
  }
  const verts = poly.slice(0, -1);
  const n = verts.length;
  const r2 = r * r;
  const EPS_T = 1e-9;

  // For each polygon edge, extract its in-disk segment (if any).
  // Each entry records: { start, end, enterFromArc, exitToArc }
  const segs = [];
  for (let i = 0; i < n; i++) {
    const a = verts[i];
    const b = verts[(i+1) % n];
    const dax = a[0] - cx, day = a[1] - cy;
    const dbx = b[0] - cx, dby = b[1] - cy;
    const aIn = (dax*dax + day*day) <= r2;
    const bIn = (dbx*dbx + dby*dby) <= r2;

    // Solve |a + t(b-a) - c|² = r²
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const AA = dx*dx + dy*dy;
    const BB = 2 * (dax*dx + day*dy);
    const CC = dax*dax + day*day - r2;
    const disc = BB*BB - 4*AA*CC;
    let t1 = NaN, t2 = NaN;
    if (disc > 0 && AA > 0) {
      const sq = Math.sqrt(disc);
      t1 = (-BB - sq) / (2*AA);
      t2 = (-BB + sq) / (2*AA);
    }
    const atT = t => [a[0] + t*dx, a[1] + t*dy];
    const inTs = (t) => (t > EPS_T && t < 1 - EPS_T);

    if (aIn && bIn) {
      // Whole edge inside
      segs.push({ start: a, end: b, enterFromArc: false, exitToArc: false });
    } else if (aIn && !bIn) {
      // Exits at the t in (0,1)
      const t = inTs(t1) ? t1 : (inTs(t2) ? t2 : NaN);
      if (!Number.isNaN(t)) segs.push({ start: a, end: atT(t), enterFromArc: false, exitToArc: true });
    } else if (!aIn && bIn) {
      // Enters at the t in (0,1)
      const t = inTs(t1) ? t1 : (inTs(t2) ? t2 : NaN);
      if (!Number.isNaN(t)) segs.push({ start: atT(t), end: b, enterFromArc: true, exitToArc: false });
    } else {
      // Both outside: edge may still cross disk at two interior t values
      if (inTs(t1) && inTs(t2)) {
        segs.push({ start: atT(t1), end: atT(t2), enterFromArc: true, exitToArc: true });
      }
      // else: edge entirely outside — skip
    }
  }

  // No straight segments at all — either the disk is entirely inside the
  // Voronoi polygon (common for isolated cells) or there is no overlap.
  if (segs.length === 0) {
    if (pointInPolygon(cx, cy, verts)) {
      const A = Math.PI * r * r, Pc = 2 * Math.PI * r;
      return { area: A, perim: Pc, arcLen: Pc, pieces: [{ type: 'fulldisk', cx, cy, r }] };
    }
    return { area: 0, perim: 0, arcLen: 0, pieces: [] };
  }

  // Walk segments in polygon order. Between segs[i] and segs[i+1]:
  //   - If segs[i].exitToArc AND segs[i+1].enterFromArc: insert arc
  //     (going CCW from exit point to enter point around disk centre)
  //   - If one of them doesn't involve an arc boundary (interior-polygon
  //     endpoint), the segments share an endpoint and no arc is needed.
  const pieces = [];
  const m = segs.length;
  for (let i = 0; i < m; i++) {
    const s = segs[i];
    pieces.push({ type: 'line', a: s.start, b: s.end });
    const nxt = segs[(i + 1) % m];
    if (s.exitToArc && nxt.enterFromArc) {
      const alpha = arcAngleCCW(s.end, nxt.start, cx, cy);
      pieces.push({ type: 'arc', a: s.end, b: nxt.start, angle: alpha, cx, cy, r });
    }
  }

  // Measure
  let A = 0, Pp = 0, Parc = 0;
  const nv = pieces.length;
  for (let i = 0; i < nv; i++) {
    const p = pieces[i];
    const pNext = pieces[(i+1) % nv];
    A += 0.5 * (p.a[0] * pNext.a[1] - pNext.a[0] * p.a[1]);
    if (p.type === 'line') {
      const dx = p.b[0] - p.a[0], dy = p.b[1] - p.a[1];
      Pp += Math.sqrt(dx*dx + dy*dy);
    } else if (p.type === 'arc') {
      A += 0.5 * r * r * (p.angle - Math.sin(p.angle));
      const arcL = r * p.angle;
      Pp += arcL;
      Parc += arcL;
    }
  }
  return { area: Math.abs(A), perim: Pp, arcLen: Parc, pieces };
}

// ══════════════════════════════════════════════════════════════
//  Voronoi construction (d3-delaunay, open BCs on a bounding box)
// ══════════════════════════════════════════════════════════════
function buildVoronoi(rxA, ryA, N, W, H) {
  const flat = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) { flat[2*i] = rxA[i]; flat[2*i+1] = ryA[i]; }
  const d = new Delaunay(flat);
  const v = d.voronoi([0, 0, W, H]);
  return { delaunay: d, voronoi: v };
}

function totalEnergyFromState(voronoi) {
  const N = P.N;
  let E = 0;
  for (let i = 0; i < N; i++) {
    const poly = voronoi.cellPolygon(i);
    const g = clipAndMeasure(poly, rx[i], ry[i], P.r);
    const dA = g.area - P.A0;
    const dPp = g.perim - P.P0;
    E += P.KA * dA * dA + P.KP * dPp * dPp + P.Lambda * g.arcLen;
  }
  return E;
}

function measureAllCells(voronoi) {
  const N = P.N;
  cellArea = new Float64Array(N);
  cellPerim = new Float64Array(N);
  cellArcLen = new Float64Array(N);
  cellPieces = new Array(N);
  for (let i = 0; i < N; i++) {
    const poly = voronoi.cellPolygon(i);
    const g = clipAndMeasure(poly, rx[i], ry[i], P.r);
    cellArea[i]   = g.area;
    cellPerim[i]  = g.perim;
    cellArcLen[i] = g.arcLen;
    cellPieces[i] = g.pieces;
  }
}

// ══════════════════════════════════════════════════════════════
//  Forces by central finite differences of the total energy
// ══════════════════════════════════════════════════════════════
function computeForces() {
  const N = P.N;
  const fx = new Float64Array(N);
  const fy = new Float64Array(N);
  const EPS = 0.005 * P.r;

  for (let i = 0; i < N; i++) {
    const origX = rx[i];
    const origY = ry[i];

    rx[i] = origX + EPS;
    const { voronoi: vp } = buildVoronoi(rx, ry, N, boxW, boxH);
    const Ep = totalEnergyFromState(vp);
    rx[i] = origX - EPS;
    const { voronoi: vm } = buildVoronoi(rx, ry, N, boxW, boxH);
    const Em = totalEnergyFromState(vm);
    rx[i] = origX;
    fx[i] = -(Ep - Em) / (2 * EPS);

    ry[i] = origY + EPS;
    const { voronoi: vp2 } = buildVoronoi(rx, ry, N, boxW, boxH);
    const Ep2 = totalEnergyFromState(vp2);
    ry[i] = origY - EPS;
    const { voronoi: vm2 } = buildVoronoi(rx, ry, N, boxW, boxH);
    const Em2 = totalEnergyFromState(vm2);
    ry[i] = origY;
    fy[i] = -(Ep2 - Em2) / (2 * EPS);
  }
  return { fx, fy };
}

// ══════════════════════════════════════════════════════════════
//  Integration step (Euler–Maruyama)
// ══════════════════════════════════════════════════════════════
function step() {
  const N = P.N;
  const { fx, fy } = computeForces();

  const dt = P.dt;
  // Cap per-step displacement so topology flips (discontinuous F) don't
  // cause large jumps. Any step longer than STEP_CAP is rescaled.
  const STEP_CAP = 0.05 * P.r;
  for (let i = 0; i < N; i++) {
    const nx = Math.cos(theta[i]);
    const ny = Math.sin(theta[i]);
    let dxi = (P.mu * fx[i] + P.v0 * nx) * dt;
    let dyi = (P.mu * fy[i] + P.v0 * ny) * dt;
    const mag = Math.hypot(dxi, dyi);
    if (mag > STEP_CAP) {
      const k = STEP_CAP / mag;
      dxi *= k; dyi *= k;
    }
    spd[i] = Math.hypot(dxi, dyi) / dt;
    dispX[i] += dxi;
    dispY[i] += dyi;
    // Soft confinement so cells don't escape the open box (gentle wall)
    let nxPos = rx[i] + dxi;
    let nyPos = ry[i] + dyi;
    const wallPad = 0.3;
    if (nxPos < wallPad)          nxPos = wallPad + 0.01;
    if (nxPos > boxW - wallPad)   nxPos = boxW - wallPad - 0.01;
    if (nyPos < wallPad)          nyPos = wallPad + 0.01;
    if (nyPos > boxH - wallPad)   nyPos = boxH - wallPad - 0.01;
    rx[i] = nxPos;
    ry[i] = nyPos;
    theta[i] += gauss() * Math.sqrt(2 * P.Dr * dt);
  }
  stepCount++;

  // Measure cells once, post-step, for rendering and stats
  const { voronoi } = buildVoronoi(rx, ry, N, boxW, boxH);
  measureAllCells(voronoi);
}

// ══════════════════════════════════════════════════════════════
//  Init — jittered hexagonal lattice, open box around it
// ══════════════════════════════════════════════════════════════
function init() {
  const N = P.N;
  rx = new Float64Array(N); ry = new Float64Array(N);
  theta = new Float64Array(N); spd = new Float64Array(N);
  dispX = new Float64Array(N); dispY = new Float64Array(N);

  const cols = Math.ceil(Math.sqrt(N));
  const rows = Math.ceil(N / cols);
  const spacing = P.spacing;
  const xStep = spacing;
  const yStep = spacing * Math.sqrt(3) / 2;
  const totalW = cols * xStep;
  const totalH = rows * yStep;
  boxW = totalW + 2 * P.boxPad;
  boxH = totalH + 2 * P.boxPad;

  let k = 0;
  for (let r = 0; r < rows && k < N; r++) {
    for (let c = 0; c < cols && k < N; c++, k++) {
      const xOff = (r % 2) * xStep / 2;
      rx[k] = P.boxPad + c * xStep + xOff + (Math.random() - 0.5) * spacing * 0.15;
      ry[k] = P.boxPad + r * yStep +          (Math.random() - 0.5) * spacing * 0.15;
      theta[k] = Math.random() * 2 * Math.PI;
    }
  }
  stepCount = 0;

  const { voronoi } = buildVoronoi(rx, ry, N, boxW, boxH);
  measureAllCells(voronoi);
}

// ══════════════════════════════════════════════════════════════
//  Message handler
// ══════════════════════════════════════════════════════════════
function postState() {
  self.postMessage({
    type: 'state',
    rx: Array.from(rx),
    ry: Array.from(ry),
    theta: Array.from(theta),
    spd: Array.from(spd),
    areas: Array.from(cellArea),
    perims: Array.from(cellPerim),
    arcLens: Array.from(cellArcLen),
    pieces: cellPieces,
    stepCount,
    boxW, boxH,
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
    for (let s = 0; s < m.nSteps; s++) step();
    postState();
  }
};
