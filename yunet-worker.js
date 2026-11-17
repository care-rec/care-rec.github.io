// もう1つの顔検出（YuNet）を、画面の動きを止めないよう別の流れ（ワーカー）で動かす。
// マスクの顔・横顔・寝ている顔に強い。映像はこのスマホの中だけで処理し、どこにも送らない。
import * as ort from './vendor/ort/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

const N = 640;          // この検出部品は 640x640 の画像を受け取る
const MIN_SCORE = 0.4;  // これ未満は返さない（判断は本体側で行う）
let session = null, cv = null, g = null, input = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      session = await ort.InferenceSession.create(m.model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      cv = new OffscreenCanvas(N, N);
      g = cv.getContext('2d', { willReadFrequently: true });
      input = new Float32Array(3 * N * N);
      await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, N, N]) });   // 慣らし
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'detect') {
      const boxes = await detect(m.bitmap, m.cells);
      self.postMessage({ type: 'result', id: m.id, boxes });
    }
  } catch (err) {
    if (m.bitmap) try { m.bitmap.close(); } catch (_) { /* なし */ }
    self.postMessage({ type: m.type === 'init' ? 'fail' : 'result', id: m.id, boxes: [], error: String(err && err.message || err) });
  }
};

// 1枚の入力画像（640x640）を区画に分け、各区画に「元の画像の (cx,cy) を中心に side 四方を rot 度回したもの」を描く。
// 画面全体を1区画で調べることも、顔の周りを4区画で一度に調べることもできる。
// 見つけた枠は、元の画像の座標に戻して、どの区画で見つけたか（cell）を付けて返す。
async function detect(bitmap, cells) {
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = '#777';
  g.fillRect(0, 0, N, N);
  for (const c of cells) {
    const a = c.rot * Math.PI / 180, k = c.size / c.side;
    g.save();
    g.beginPath();
    g.rect(c.ox, c.oy, c.size, c.size);
    g.clip();
    g.translate(c.ox + c.size / 2, c.oy + c.size / 2);
    g.rotate(a);
    g.scale(k, k);
    g.translate(-c.cx, -c.cy);
    g.drawImage(bitmap, 0, 0);
    g.restore();
  }
  bitmap.close();
  const d = g.getImageData(0, 0, N, N).data;
  const P = N * N;
  for (let i = 0; i < P; i++) { input[i] = d[i * 4 + 2]; input[P + i] = d[i * 4 + 1]; input[2 * P + i] = d[i * 4]; }   // BGRの順
  const o = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, N, N]) });
  const found = [];
  for (const s of [8, 16, 32]) {
    const cols = N / s, rows = N / s;
    const cls = o['cls_' + s].data, obj = o['obj_' + s].data, bb = o['bbox_' + s].data, kp = o['kps_' + s].data;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const sc = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
      if (sc < MIN_SCORE) continue;
      const cx = (c + bb[i * 4]) * s, cy = (r + bb[i * 4 + 1]) * s;
      const bw = Math.exp(bb[i * 4 + 2]) * s, bh = Math.exp(bb[i * 4 + 3]) * s;
      const pts = [];
      for (let k = 0; k < 5; k++) pts.push([(c + kp[i * 10 + k * 2]) * s, (r + kp[i * 10 + k * 2 + 1]) * s]);
      found.push({ cx, cy, bw, bh, s: sc, pts });
    }
  }
  found.sort((p, q) => q.s - p.s);
  const keep = [];
  for (const f of found) if (!keep.some((k) => iou(k, f) > 0.3)) keep.push(f);
  const out = [];
  for (const f of keep) {
    const ci = cells.findIndex((c) => f.cx >= c.ox && f.cx < c.ox + c.size && f.cy >= c.oy && f.cy < c.oy + c.size);
    if (ci < 0) continue;
    const c = cells[ci];
    // 区画からはみ出す大きな枠は、となりの区画とまたがった誤検出なので捨てる
    if (cells.length > 1 && (f.bw > c.size * 0.95 || f.bh > c.size * 0.95)) continue;
    const a = c.rot * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a), k = c.size / c.side;
    const back = (u, v) => {
      const dx = (u - c.ox - c.size / 2) / k, dy = (v - c.oy - c.size / 2) / k;
      return { x: c.cx + dx * ca + dy * sa, y: c.cy - dx * sa + dy * ca };
    };
    const p = back(f.cx, f.cy);
    // 回転した画像での縦横を、元の画像での縦横に直す（90度に近いときは入れかえる）
    const turned = Math.abs(sa) > Math.abs(ca);
    const w = (turned ? f.bh : f.bw) / k, h = (turned ? f.bw : f.bh) / k;
    // 斜め（45度など）のときは、顔がはみ出さないよう外側を囲む大きさにする
    const diag = Math.abs(Math.abs(sa) - Math.abs(ca)) < 0.5;
    const ww = diag ? Math.max(w, h) * 1.12 : w, hh = diag ? Math.max(w, h) * 1.12 : h;
    out.push({ x: p.x - ww / 2, y: p.y - hh / 2, w: ww, h: hh, s: f.s, rot: c.rot, cell: ci, kp: f.pts.map(([u, v]) => back(u, v)) });
  }
  return out;
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
function iou(a, b) {
  const x0 = Math.max(a.cx - a.bw / 2, b.cx - b.bw / 2), y0 = Math.max(a.cy - a.bh / 2, b.cy - b.bh / 2);
  const x1 = Math.min(a.cx + a.bw / 2, b.cx + b.bw / 2), y1 = Math.min(a.cy + a.bh / 2, b.cy + b.bh / 2);
  const i = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return i / (a.bw * a.bh + b.bw * b.bh - i || 1);
}
