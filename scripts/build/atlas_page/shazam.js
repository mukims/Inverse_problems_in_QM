// Shazam's label-free lookup in the browser. Mirrors atlaslib: InputSpec.to_input (with v4 despiking),
// the Conv1dAE encoder (conv -> GroupNorm(1, C) -> ReLU, x3; linear), and Atlas.locate (k-NN vote, class-conditional novelty).

const round3 = (v) => Math.round(v * 1000) / 1000;

export function despike(T) {
  const n = T.length, out = Float64Array.from(T), w = [0, 0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let j = -2; j <= 2; j++) w[j + 2] = T[Math.min(n - 1, Math.max(0, i + j))];
    const m = [...w].sort((a, b) => a - b)[2];
    if (T[i] > 2 * m + 2) out[i] = m;
  }
  return out;
}

function interp(x, xs, ys) {                       // np.interp: clamps outside [xs[0], xs[last]]
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
}

export function toInput(E, T, spec, bandTop = null) {
  if (E.length !== T.length) throw new Error(`T has ${T.length} channels but E has ${E.length}`);
  if (Math.abs(E[0]) > 1e-9 || E.some((e, i) => i > 0 && e <= E[i - 1])) throw new Error("energies must start at 0 and increase");
  const t = spec.despike ? despike(Array.from(T, round3)) : Array.from(T);   // Python rounds before interpolating only when despiking
  const n = Math.round(spec.e_max_t / spec.step_t), step = spec.step_t;
  const grid = Array.from({ length: n }, (_, i) => i * step);
  const last = E[E.length - 1];
  let inside;
  if (bandTop === null || bandTop === undefined) {
    if (last < grid[n - 1] - step / 2) throw new Error(`data ends at ${last.toFixed(3)} ${spec.unit} but window extends to ${grid[n - 1].toFixed(3)} ${spec.unit}`);
    inside = grid.map(() => true);
  } else {
    inside = grid.map((g) => g < bandTop - 1e-9);
    const k = inside.lastIndexOf(true), required = k >= 0 ? grid[k] : 0;
    if (last < required - step / 2) throw new Error(`data ends at ${last.toFixed(3)} ${spec.unit} but the band extends to ${bandTop.toFixed(3)} ${spec.unit}; zero-filling would erase real signal`);
  }
  const out = new Float32Array(n), lc = Math.log1p(spec.cap);
  for (let i = 0; i < n; i++) {
    const v = inside[i] ? Math.min(Math.max(round3(interp(grid[i], E, t)), 0), spec.cap) : 0;
    out[i] = Math.log1p(v) / lc;
  }
  return out;
}

function conv1d(x, Cin, L, W, b, Cout, K, stride, pad) {
  const Lout = Math.floor((L + 2 * pad - K) / stride) + 1, y = new Float64Array(Cout * Lout);
  for (let o = 0; o < Cout; o++) for (let t = 0; t < Lout; t++) {
    let s = b[o];
    for (let c = 0; c < Cin; c++) for (let k = 0; k < K; k++) {
      const i = t * stride - pad + k;
      if (i >= 0 && i < L) s += W[(o * Cin + c) * K + k] * x[c * L + i];
    }
    y[o * Lout + t] = s;
  }
  return [y, Lout];
}

function groupNormReLU(y, C, L, gamma, beta) {     // GroupNorm(1, C): one group over all C*L values, biased variance
  let mean = 0; for (let i = 0; i < y.length; i++) mean += y[i]; mean /= y.length;
  let v = 0; for (let i = 0; i < y.length; i++) v += (y[i] - mean) ** 2; v /= y.length;
  const inv = 1 / Math.sqrt(v + 1e-5);
  for (let c = 0; c < C; c++) for (let t = 0; t < L; t++) {
    const i = c * L + t, h = (y[i] - mean) * inv * gamma[c] + beta[c];
    y[i] = h > 0 ? h : 0;
  }
  return y;
}

export function encode(x, enc) {
  let h = x, C = 1, L = x.length;
  for (const l of enc.layers) {
    const [y, Lout] = conv1d(h, C, L, l.W, l.b, l.Cout, l.K, l.stride, l.pad);
    h = groupNormReLU(y, l.Cout, Lout, l.gamma, l.beta);
    C = l.Cout; L = Lout;
  }
  const z = new Float64Array(enc.latent);
  for (let j = 0; j < enc.latent; j++) {
    let s = enc.linB[j];
    for (let i = 0; i < h.length; i++) s += enc.linW[j * h.length + i] * h[i];
    z[j] = s;
  }
  return z;
}

export async function loadPage(baseUrl, fetchFn = fetch) {
  const get = (name) => fetchFn(baseUrl + name);
  const meta = await (await get("model.json")).json();
  const f32 = async (name) => new Float32Array(await (await get(name)).arrayBuffer());
  const [refs, refDensity, encFlat] = await Promise.all([f32("refs.bin"), f32("ref_density.bin"), f32("encoder.bin")]);
  const refModel = new Uint16Array(await (await get("ref_model.bin")).arrayBuffer());
  const D = meta.mu.length, layers = [];
  let off = 0, Cin = 1;
  const take = (n) => { const a = encFlat.subarray(off, off + n); off += n; return a; };
  for (const c of meta.encoder.convs) {
    layers.push({ ...c, W: take(c.Cout * Cin * c.K), b: take(c.Cout), gamma: take(c.Cout), beta: take(c.Cout) });
    Cin = c.Cout;
  }
  const flat = (encFlat.length - off - D) / D;
  const enc = { layers, latent: D, linW: take(D * flat), linB: take(D) };
  const spec = { ...meta.spec, n_channels: Math.round(meta.spec.e_max_t / meta.spec.step_t) };
  return { ...meta, spec, enc, D, refs, refModel, refDensity };
}

function nearest(zs, page, filter, k) {             // k smallest Euclidean distances, ascending
  const best = [];
  for (let r = 0; r < page.n_refs; r++) {
    if (filter !== null && page.refModel[r] !== filter) continue;
    let d = 0;
    for (let j = 0; j < page.D; j++) { const u = zs[j] - page.refs[r * page.D + j]; d += u * u; }
    d = Math.sqrt(d);
    if (best.length < k || d < best[best.length - 1][0]) {
      best.push([d, r]); best.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      if (best.length > k) best.pop();
    }
  }
  return best;
}

const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

export function locate(x, page) {
  const z = encode(x, page.enc), zs = new Float64Array(page.D);
  for (let j = 0; j < page.D; j++) zs[j] = (z[j] - page.mu[j]) / page.sd[j];
  const nb = nearest(zs, page, null, page.k);
  const groups = new Map();                          // first appearance in neighbour order breaks ties, as dict order in Python
  nb.forEach(([, r], j) => {
    const m = page.models[page.refModel[r]], key = `${m.material}|${m.edge}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(j);
  });
  let topKey = null, members = [];
  for (const [key, js] of groups) if (js.length > members.length) { topKey = key; members = js; }
  const [material, edge] = topKey.split("|");
  const w = members.map((j) => page.models[page.refModel[nb[j][1]]].width);
  const wt = members.map((j) => 1 / (nb[j][0] + 1e-9));
  const trained = page.models.filter((m) => m.material === material && m.edge === edge).map((m) => m.width).sort((a, b) => a - b);
  let width = w.reduce((s, v, i) => s + v * wt[i], 0) / wt.reduce((s, v) => s + v, 0);
  width = Math.min(Math.max(width, trained[0]), trained[trained.length - 1]);
  const counts = new Map(); w.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
  const widthVote = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];   // smallest width among ties
  const density = median(members.map((j) => page.refDensity[nb[j][1]]));
  let target = page.models.findIndex((m) => m.material === material && m.edge === edge && Math.round(m.width) === Math.round(widthVote));
  if (target < 0) {
    const mc = new Map(); members.forEach((j) => { const i = page.refModel[nb[j][1]]; mc.set(i, (mc.get(i) || 0) + 1); });
    target = [...mc.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
  }
  const nearestModel = page.models[target].id;
  const own = nearest(zs, page, target, page.k);
  const s = own.reduce((acc, [d]) => acc + d, 0) / own.length;
  const table = page.threshold_table[nearestModel];
  let unknown = false, ratio = 0, zscore = null;
  if (table) {
    let snapped = null, bestGap = Infinity;
    for (const key of Object.keys(table)) { const gap = Math.abs(parseFloat(key) - density); if (gap < bestGap) { bestGap = gap; snapped = key; } }
    const tau = table[snapped];
    unknown = s > tau; ratio = s / tau;
    const p = (page.threshold_params[nearestModel] || {})[snapped];
    if (p) zscore = (Math.log(Math.max(s, 1e-12)) - p.c) / p.w;
  }
  return { material, edge, width, width_vote: widthVote, density, unknown, novelty_ratio: ratio, novelty_z: zscore,
           nearest_model: nearestModel, zs: Array.from(zs), neighbours: nb.map(([d, r]) => [page.models[page.refModel[r]].id, d]) };
}

export function project(zs, page) {
  const [c1, c2] = page.pca.components, m = page.pca.mean;
  let x = 0, y = 0;
  for (let j = 0; j < page.D; j++) { x += (zs[j] - m[j]) * c1[j]; y += (zs[j] - m[j]) * c2[j]; }
  return [x, y];
}
