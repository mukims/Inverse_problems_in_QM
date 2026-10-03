// node test_shazam.mjs <data_dir> : compares shazam.js with the Python answers in test_vectors.json
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { despike, toInput, encode, loadPage, locate } from "./shazam.js";

const dir = process.argv[2];
const fetchFile = async (name) => {
  const buf = readFileSync(join(dir, name));
  return { json: async () => JSON.parse(buf.toString()), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const page = await loadPage("", (url) => fetchFile(url));
const vectors = JSON.parse(readFileSync(join(dir, "test_vectors.json")).toString());
let maxX = 0, maxZ = 0;
const agree = { material: 0, edge: 0, width_vote: 0, unknown: 0, nearest_model: 0 };
for (const v of vectors) {
  const x = toInput(v.e_ev, v.T, page.spec, v.band_top_ev);
  x.forEach((xi, i) => { maxX = Math.max(maxX, Math.abs(xi - v.x[i])); });
  const a = locate(x, page);
  a.zs.forEach((zi, i) => { maxZ = Math.max(maxZ, Math.abs(zi - v.zs[i])); });
  for (const f of Object.keys(agree)) agree[f] += a[f] === v.answer[f] ? 1 : 0;
}
const pct = Object.fromEntries(Object.entries(agree).map(([f, n]) => [f, (100 * n) / vectors.length]));
console.log(JSON.stringify({ n: vectors.length, maxX, maxZ, pct }));

// despike keeps a +2 plateau and removes a lone spike
const kept = despike([6, 6, 6, 8, 8, 6, 6, 6]), cut = despike([0, 0, 461.7, 0.147, 0, 0]);
const ok = maxX < 2e-3 && maxZ < 1e-3 && Object.values(pct).every((p) => p >= 99.5)
  && kept[3] === 8 && cut[2] === 0 && Math.abs(cut[3] - 0.147) < 1e-9;
// refuses short data without a band top; clamps like np.interp
let refused = false;
try { toInput([0, 1, 2], [1, 1, 1], page.spec, null); } catch (e) { refused = /window extends/.test(e.message); }
const coarse = toInput([0, 4.16, 8.32], [2, 2, 2], page.spec, null);
const clampOk = Math.abs(coarse[page.spec.n_channels - 1] - Math.log1p(2) / Math.log1p(page.spec.cap)) < 1e-6;
if (!(ok && refused && clampOk)) { console.error("FAIL", { ok, refused, clampOk }); process.exit(1); }
console.log("PASS");
