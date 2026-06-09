import { initVoro, energyAndForce, relax, makeConfig } from './voro3d.js';
await initVoro();
const N=40,s0=5.2,kV=1,kS=1,seed=7; const cr=makeConfig(N,seed);
const r=energyAndForce(cr,N,s0,kV,kS);
for(let w=0;w<3;w++) energyAndForce(cr,N,s0,kV,kS);
let t=Date.now();const M=30;for(let q=0;q<M;q++)energyAndForce(cr,N,s0,kV,kS);
let sV=0;for(let i=0;i<N;i++)sV+=r.Vc[i];
console.log(`N=40 eval ${((Date.now()-t)/M).toFixed(2)}ms  E/N=${r.EperN.toExponential(3)}  sum(Vc)=${sV.toFixed(6)} (want 1)`);
const Ef=c=>energyAndForce(c,N,s0,kV,kS).E;const h=1e-6;let fderr=0;
for(const k of [0,5,11,23,40,61,77,99,118].filter(k=>k<3*N)){const cp=Float64Array.from(cr);cp[k]+=h;const cm=Float64Array.from(cr);cm[k]-=h;fderr=Math.max(fderr,Math.abs(-(Ef(cp)-Ef(cm))/(2*h)-r.F[k]));}
let sx=0,sy=0,sz=0;for(let i=0;i<N;i++){sx+=r.F[3*i];sy+=r.F[3*i+1];sz+=r.F[3*i+2];}
console.log(`FD rel=${(fderr/Math.max(r.maxF,1)).toExponential(3)}  momentum=${(Math.abs(sx)+Math.abs(sy)+Math.abs(sz)).toExponential(2)}`);
const ev=relax(60,5.0,1,0,3,{ftol:1e-6,maxSteps:1500});
const V=ev.Vc,mean=V.reduce((a,b)=>a+b,0)/V.length,sd=Math.sqrt(V.reduce((a,b)=>a+(b-mean)**2,0)/V.length);
console.log(`equal-vol(kS=0): status=${ev.status} evals=${ev.nevals} CoV=${(sd/mean).toExponential(3)} maxF=${ev.maxF.toExponential(2)}`);
const c2=makeConfig(96,1);for(let w=0;w<3;w++)energyAndForce(c2,96,5.2,1,1);
t=Date.now();for(let q=0;q<20;q++)energyAndForce(c2,96,5.2,1,1);console.log(`N=96 eval ${((Date.now()-t)/20).toFixed(2)}ms`);
console.log('V_DONE');
