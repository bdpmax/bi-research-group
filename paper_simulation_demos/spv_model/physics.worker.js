// ══════════════════════════════════════════════════════════════
//  Self-Propelled Voronoi physics worker (classic, importScripts)
//  - Uses d3-delaunay UMD for O(n log n) triangulation
//  - Periodic BCs via 3×3 ghost tiling
//  - Analytical forces via cellGPU circumcenter-Jacobian chain rule
//  - Integrates Euler–Maruyama, tracks unwrapped displacement
//  - Logs MSD(t) samples (20 / decade, log-spaced) when requested
// ══════════════════════════════════════════════════════════════
importScripts('https://cdn.jsdelivr.net/npm/d3-delaunay@6/dist/d3-delaunay.min.js');
const { Delaunay } = self.d3;

// ── Parameters & state ────────────────────────────────────────
let P = { N: 64, p0: 3.80, v0: 0.05, Dr: 1.00, KA: 1.0, KP: 1.0, dt: 0.02, mu: 1.0 };
let rx, ry, th, spd;
let dispX, dispY;                    // unwrapped displacement accumulator
let cellArea, cellPeri;
let voroList;                        // only voroList is exposed to main (for rendering)
let stepCount = 0;

// Short-time Deff (running window of DEFF_INT steps)
const DEFF_INT = 100;
let defRefX, defRefY, defRefStep;
let Deff = null;

// MSD log-log samples
let msdMode = false;
let msdRefStep, msdRefX, msdRefY;
let msdSamples = [];
let msdNextLog = -Infinity;
const MSD_DECADES_SAMPLES = 20;
const MSD_MAX = 400;

// ── Helpers ────────────────────────────────────────────────────
const wrap = x => ((x % 1) + 1) % 1;
function minDisp(ax, ay, bx, by) {
  let dx = bx - ax, dy = by - ay;
  if (dx >  0.5) dx -= 1; if (dx < -0.5) dx += 1;
  if (dy >  0.5) dy -= 1; if (dy < -0.5) dy += 1;
  return [dx, dy];
}
let _g = null;
function gauss() {
  if (_g !== null) { const v = _g; _g = null; return v; }
  let u, v, s;
  do { u = Math.random()*2-1; v = Math.random()*2-1; s = u*u+v*v; } while (s>=1 || s===0);
  const f = Math.sqrt(-2*Math.log(s)/s); _g = v*f; return u*f;
}
function circumcircle(ax,ay,bx,by,cx,cy) {
  const D = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by));
  if (Math.abs(D) < 1e-14) return null;
  const ux = ((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/D;
  const uy = ((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/D;
  return { ux, uy };
}

// ── Init (Poisson on [0,1)²) ───────────────────────────────────
function init() {
  const N = P.N;
  rx = new Float64Array(N); ry = new Float64Array(N);
  th = new Float64Array(N); spd = new Float64Array(N);
  dispX = new Float64Array(N); dispY = new Float64Array(N);
  cellArea = new Float64Array(N); cellPeri = new Float64Array(N);

  for (let k = 0; k < N; k++) {
    rx[k] = Math.random();
    ry[k] = Math.random();
    th[k] = Math.random() * 2 * Math.PI;
  }
  stepCount = 0;

  defRefX = new Float64Array(N); defRefY = new Float64Array(N);
  defRefStep = 0;
  Deff = null;

  resetMSDLog();
  voroList = null;
}

function resetMSDLog() {
  if (!dispX) return;
  msdRefStep = stepCount;
  msdRefX = dispX.slice();
  msdRefY = dispY.slice();
  msdSamples = [];
  msdNextLog = -Infinity;
}

// ══════════════════════════════════════════════════════════════
//  Delaunay via d3-delaunay with 3×3 PBC tiling
// ══════════════════════════════════════════════════════════════
function buildDelaunay(rxA, ryA, N) {
  const flat = new Float64Array(9 * N * 2);
  const origs = new Int32Array(9 * N);
  let idx = 0;
  for (let di = -1; di <= 1; di++)
    for (let dj = -1; dj <= 1; dj++)
      for (let k = 0; k < N; k++) {
        flat[idx*2]   = rxA[k] + dj;
        flat[idx*2+1] = ryA[k] + di;
        origs[idx]    = k;
        idx++;
      }
  const centralStart = 4 * N;
  const delaunay = new Delaunay(flat);
  const tris = delaunay.triangles;
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

function buildNeighboursAndVoro(rxA, ryA, N, tris) {
  const triOfCell = Array.from({length:N}, () => []);
  for (const [a,b,c] of tris) {
    for (const [self, n1, n2] of [[a,b,c],[b,a,c],[c,a,b]]) {
      const sx = rxA[self], sy = ryA[self];
      const [d1x,d1y] = minDisp(sx, sy, rxA[n1], ryA[n1]);
      const [d2x,d2y] = minDisp(sx, sy, rxA[n2], ryA[n2]);
      const bx = sx+d1x, by = sy+d1y;
      const cx2 = sx+d2x, cy2 = sy+d2y;
      const cc = circumcircle(sx, sy, bx, by, cx2, cy2);
      if (!cc) continue;
      triOfCell[self].push({vx: cc.ux, vy: cc.uy, n1, n2});
    }
  }
  const nbrsList = new Array(N);
  const voroList = new Array(N);
  for (let i = 0; i < N; i++) {
    const ts = triOfCell[i];
    if (ts.length < 3) { nbrsList[i]=[]; voroList[i]=[]; continue; }
    const sx = rxA[i], sy = ryA[i];
    ts.sort((ta,tb) => Math.atan2(ta.vy-sy, ta.vx-sx) - Math.atan2(tb.vy-sy, tb.vx-sx));
    const nv = ts.length;
    const nOrder = new Array(nv);
    for (let m = 0; m < nv; m++) {
      const tm = ts[m];
      const tnext = ts[(m+1)%nv];
      const tmSet = new Set([tm.n1, tm.n2]);
      let shared = -1;
      for (const x of [tnext.n1, tnext.n2]) if (tmSet.has(x)) { shared = x; break; }
      nOrder[m] = shared;
    }
    nbrsList[i] = nOrder;
    voroList[i] = ts.map(t => ({x: t.vx, y: t.vy}));
  }
  return { nbrs: nbrsList, voroVerts: voroList };
}

function computeGeometry(rxA, ryA, N, nbrsList, voroList) {
  const areas = new Float64Array(N);
  const peris = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const vs = voroList[i];
    const nv = vs.length;
    if (nv < 3) continue;
    let A = 0, Pe = 0;
    for (let m = 0; m < nv; m++) {
      const v0 = vs[m], v1 = vs[(m+1)%nv];
      A += v0.x*v1.y - v1.x*v0.y;
      const dx = v1.x-v0.x, dy = v1.y-v0.y;
      Pe += Math.sqrt(dx*dx + dy*dy);
    }
    areas[i] = Math.abs(A) * 0.5;
    peris[i] = Pe;
  }
  return { areas, peris };
}

// ══════════════════════════════════════════════════════════════
//  Force computation — cellGPU circumcenter-Jacobian chain rule
// ══════════════════════════════════════════════════════════════
const THRESH = 1e-12;
function computeForces(rxA, ryA, N, A0, P0, nbrsList, voroList, areas, peris) {
  const fx = new Float64Array(N);
  const fy = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    const ns = nbrsList[i];
    const voro = voroList[i];
    const neigh = ns.length;
    if (neigh < 3) continue;

    const pix = rxA[i], piy = ryA[i];
    const dhdri = new Array(neigh);
    let [rijx, rijy] = minDisp(pix, piy, rxA[ns[neigh-1]], ryA[ns[neigh-1]]);

    for (let nn = 0; nn < neigh; nn++) {
      const [rikx, riky] = minDisp(pix, piy, rxA[ns[nn]], ryA[ns[nn]]);
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
      const vnext = voro[(nn+1)%neigh];
      const baseNeigh  = ns[nn];
      const otherNeigh = ns[(nn-1+neigh)%neigh];

      const dAidvx = 0.5*(vlast.y - vnext.y);
      const dAidvy = 0.5*(vnext.x - vlast.x);
      let dlx = vlast.x - vcur.x, dly = vlast.y - vcur.y;
      let dnx = vcur.x - vnext.x, dny = vcur.y - vnext.y;
      const dlnorm = Math.max(Math.sqrt(dlx*dlx + dly*dly), THRESH);
      const dnnorm = Math.max(Math.sqrt(dnx*dnx + dny*dny), THRESH);
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
      const vcur_px = ((vcur.x % 1) + 1) % 1;
      const vcur_py = ((vcur.y % 1) + 1) % 1;
      const vk_px   = ((vk_raw.x % 1) + 1) % 1;
      const vk_py   = ((vk_raw.y % 1) + 1) % 1;
      let dvox = vk_px - vcur_px, dvoy = vk_py - vcur_py;
      if (dvox >  0.5) dvox -= 1;
      if (dvox < -0.5) dvox += 1;
      if (dvoy >  0.5) dvoy -= 1;
      if (dvoy < -0.5) dvoy += 1;
      const vox = vcur.x + dvox;
      const voy = vcur.y + dvoy;

      const dAkdvx = 0.5*(vnext.y - voy);
      const dAkdvy = 0.5*(vox - vnext.x);
      dlx = vnext.x - vcur.x; dly = vnext.y - vcur.y;
      dnx = vcur.x - vox;     dny = vcur.y - voy;
      const dlkn = Math.max(Math.sqrt(dlx*dlx + dly*dly), THRESH);
      const dnkn = Math.max(Math.sqrt(dnx*dnx + dny*dny), THRESH);
      const dPkdvx = dlx/dlkn - dnx/dnkn;
      const dPkdvy = dly/dlkn - dny/dnkn;

      const dAjdvx = 0.5*(voy - vlast.y);
      const dAjdvy = 0.5*(vlast.x - vox);
      dlx = vox - vcur.x;     dly = voy - vcur.y;
      dnx = vcur.x - vlast.x; dny = vcur.y - vlast.y;
      const dljn = Math.max(Math.sqrt(dlx*dlx + dly*dly), THRESH);
      const dnjn = Math.max(Math.sqrt(dnx*dnx + dny*dny), THRESH);
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
    fx[i] = fsx;
    fy[i] = fsy;
  }
  return { fx, fy };
}

// ══════════════════════════════════════════════════════════════
//  Integration step
// ══════════════════════════════════════════════════════════════
function step() {
  const N = P.N;
  const A0 = 1.0 / N;
  const P0 = P.p0 * Math.sqrt(A0);

  const tris = buildDelaunay(rx, ry, N);
  const { nbrs: nbs, voroVerts: vv } = buildNeighboursAndVoro(rx, ry, N, tris);
  voroList = vv;
  const { areas, peris } = computeGeometry(rx, ry, N, nbs, vv);
  cellArea = areas; cellPeri = peris;
  const { fx, fy } = computeForces(rx, ry, N, A0, P0, nbs, vv, areas, peris);

  const dt = P.dt;
  for (let i = 0; i < N; i++) {
    const nx = Math.cos(th[i]);
    const ny = Math.sin(th[i]);
    const dxi = (P.mu * fx[i] + P.v0 * nx) * dt;
    const dyi = (P.mu * fy[i] + P.v0 * ny) * dt;
    spd[i] = Math.hypot(dxi, dyi) / dt;
    dispX[i] += dxi;
    dispY[i] += dyi;
    rx[i] = wrap(rx[i] + dxi);
    ry[i] = wrap(ry[i] + dyi);
    th[i] += gauss() * Math.sqrt(2 * P.Dr * dt);
  }
  stepCount++;

  // Running Deff
  if (stepCount > 0 && stepCount % DEFF_INT === 0) {
    let msd = 0;
    for (let i = 0; i < N; i++) {
      const dx = dispX[i] - defRefX[i];
      const dy = dispY[i] - defRefY[i];
      msd += dx*dx + dy*dy;
    }
    msd /= N;
    const dt2 = (stepCount - defRefStep) * P.dt;
    Deff = msd / (4 * dt2);
    defRefX = dispX.slice();
    defRefY = dispY.slice();
    defRefStep = stepCount;
  }

  // MSD log-log samples
  if (msdMode) {
    const stepsSince = stepCount - msdRefStep;
    if (stepsSince >= 1) {
      const logS = Math.log10(stepsSince);
      if (logS >= msdNextLog) {
        let msd = 0;
        for (let i = 0; i < N; i++) {
          const dx = dispX[i] - msdRefX[i];
          const dy = dispY[i] - msdRefY[i];
          msd += dx*dx + dy*dy;
        }
        msd /= N;
        msdSamples.push({ t: stepsSince * P.dt, msd });
        if (msdSamples.length > MSD_MAX) msdSamples.shift();
        msdNextLog = logS + 1.0 / MSD_DECADES_SAMPLES;
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════
//  Message handler
// ══════════════════════════════════════════════════════════════
function postState() {
  self.postMessage({
    type: 'state',
    rx: Array.from(rx),
    ry: Array.from(ry),
    th: Array.from(th),
    spd: Array.from(spd),
    areas: Array.from(cellArea),
    peris: Array.from(cellPeri),
    voroVerts: voroList,
    stepCount,
    Deff,
    msdSamples: msdMode ? msdSamples.slice() : null,
  });
}

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'init') {
    if (msg.params) Object.assign(P, msg.params);
    init();
    postState();
  } else if (msg.type === 'setParams') {
    Object.assign(P, msg.params);
  } else if (msg.type === 'step') {
    for (let s = 0; s < msg.nSteps; s++) step();
    postState();
  } else if (msg.type === 'setMSDMode') {
    msdMode = msg.enabled;
    if (msdMode) resetMSDLog();
  }
};
