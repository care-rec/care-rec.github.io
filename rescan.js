// 撮影後の探し直し。録った動画（自動ぼかし済み）を、少しずつ場面を進めながら時間をかけて調べ、
// ぼかしが入っていない顔を見つけたら、その時間だけ手動ぼかしの候補（だ円）として返す。
// - 撮影中より念入りに調べる（顔の向きを4通りに回して調べる）
// - すでにぼかしが入っている所は、細かい模様が無いので候補にしない
// - 撮影中に「ぼかしなし」にした人の位置（clearLog）も候補にしない
// 映像はこのスマホの中だけで処理し、どこにも送らない。結果は位置と時刻だけ。
import { YuNetDetector } from './yunet.js';
import { seekFrame } from './track.js';

const STEP = 0.3;            // 何秒ごとの場面を調べるか
const Y_OK = 0.62;           // まっすぐの向きの合格点
const Y_OK_ROT = 0.72;       // 回した向きの合格点
const B_OK = 0.7;            // BlazeFace の合格点
const SURE = 0.82;           // 1回だけ見つかった顔は、この点数以上のときだけ候補にする
const LINK_GAP = 1.0;        // この秒数より間があいたら、別の候補にする
const BLURRED = 2.0;          // 細かさがこれ未満なら、すでにぼかしが入っているとみなす（はっきりぼけている所だけ。迷う所は候補に残す）
const GROW_W = 1.5, GROW_H = 1.6;   // 見つけた顔の枠に対する、ぼかしの大きさ

// 顔の枠の中の「細かさ」（となりの画素との明るさの差の平均）。ぼかし済みの所は小さい
const tiny = document.createElement('canvas');
tiny.width = 32; tiny.height = 32;
const tctx = tiny.getContext('2d', { willReadFrequently: true });
export function detail(src, b) {
  const x = b.x + b.w * 0.2, y = b.y + b.h * 0.2, w = b.w * 0.6, h = b.h * 0.6;
  tctx.drawImage(src, x, y, w, h, 0, 0, 32, 32);
  const d = tctx.getImageData(0, 0, 32, 32).data;
  let s = 0, n = 0;
  for (let j = 0; j < 32; j++) {
    for (let i = 0; i < 32; i++) {
      const a = (j * 32 + i) * 4;
      const g = d[a] * 0.3 + d[a + 1] * 0.59 + d[a + 2] * 0.11;
      if (i) { const p = a - 4; s += Math.abs(g - (d[p] * 0.3 + d[p + 1] * 0.59 + d[p + 2] * 0.11)); n++; }
      if (j) { const p = a - 128; s += Math.abs(g - (d[p] * 0.3 + d[p + 1] * 0.59 + d[p + 2] * 0.11)); n++; }
    }
  }
  return s / Math.max(1, n);
}

const center = (b) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });

function nms(list) {
  list.sort((a, b) => b.s - a.s);
  const out = [];
  for (const b of list) {
    const c = center(b);
    if (out.some((o) => { const q = center(o); return Math.hypot(c.x - q.x, c.y - q.y) < Math.max(o.w, o.h, b.w, b.h) * 0.5; })) continue;
    out.push(b);
  }
  return out;
}

// 「ぼかしなし」にした人と重なっていれば true
function isCleared(b, t, log, W, H) {
  const c = center(b);
  for (const e of log) {
    if (Math.abs(e.t - t) > 0.6) continue;
    const kx = W / (e.W || W), ky = H / (e.H || H);
    for (const r of e.r) {
      const x = r.x * kx, y = r.y * ky, w = r.w * kx, h = r.h * ky;
      if (c.x > x - w * 0.15 && c.x < x + w * 1.15 && c.y > y - h * 0.15 && c.y < y + h * 1.15) return true;
    }
  }
  return false;
}

// blob：調べる動画。camera：撮影で使った BlurCamera（顔検出の部品を借りる）。
// 戻り値 { cancel(), done: Promise<候補の一覧> }。候補は手動ぼかしと同じ形 { from, to, keys }
export function findMissedFaces(blob, { camera, clearLog = [], durationSec = 0, onProgress } = {}) {
  const url = URL.createObjectURL(blob);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  v.src = url;
  // 押した直後に一度再生しておく（iPhoneでは、こうしないと場面を移れないことがある）
  const unlock = v.play().then(() => v.pause()).catch(() => {});
  let cancelled = false;
  const job = { cancel() { cancelled = true; }, done: null };
  job.done = (async () => {
    let yunet = null, ownY = false;
    try {
      await unlock;
      if (!v.videoWidth) await new Promise((r) => { v.addEventListener('loadedmetadata', r, { once: true }); setTimeout(r, 5000); });
      const W = v.videoWidth, H = v.videoHeight;
      if (!W || !H) throw new Error('動画を読み込めませんでした');
      const dur = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : durationSec;
      // 顔検出の部品：撮影で使ったものを借りる。無ければここで読み込む
      yunet = camera && camera.yunet;
      if (!yunet) { yunet = new YuNetDetector(); ownY = await yunet.load(); if (!ownY) yunet = null; }
      const blaze = camera && camera.detector;
      if (!yunet && !blaze) throw new Error('顔検出の部品を読み込めませんでした');
      const c = document.createElement('canvas'); c.width = W; c.height = H;
      const g = c.getContext('2d', { willReadFrequently: true });
      const flip = document.createElement('canvas'); flip.width = W; flip.height = H;
      const fg = flip.getContext('2d');
      const seek = (t) => seekFrame(v, t);
      const samples = [];
      for (let t = 0.05; t < dur - 0.02; t += STEP) {
        if (cancelled) throw new Error('cancelled');
        await seek(t);
        g.drawImage(v, 0, 0, W, H);
        const found = [];
        if (yunet) {
          for (const rot of [0, 90, 270, 180]) {
            const bs = await yunet.detect(c, [YuNetDetector.fullCell(W, H, rot)]);
            for (const b of bs) if (b.s >= (rot ? Y_OK_ROT : Y_OK)) found.push({ x: b.x, y: b.y, w: b.w, h: b.h, s: b.s, rot });
          }
        }
        if (blaze) {
          try {
            fg.setTransform(-1, 0, 0, -1, W, H); fg.drawImage(c, 0, 0); fg.setTransform(1, 0, 0, 1, 0, 0);
            for (const [src, up] of [[c, true], [flip, false]]) {
              for (const d of blaze.detectForVideo(src, camera._ts(performance.now())).detections) {
                const b = d.boundingBox, s = d.categories?.[0]?.score ?? 0;
                if (!b || s < B_OK) continue;
                const x = up ? b.originX : W - b.originX - b.width, y = up ? b.originY : H - b.originY - b.height;
                found.push({ x, y, w: b.width, h: b.height, s, rot: up ? 0 : 180 });
              }
            }
          } catch (e) { /* BlazeFace が使えなくても YuNet だけで続ける */ }
        }
        const faces = [];
        for (const b of nms(found)) {
          if (b.w < 12 || b.h < 12) continue;
          if (b.w > W * 0.75 || b.h > H * 0.75) continue;   // 画面いっぱいの枠は顔ではない
          const dt = detail(c, b);
          if (globalThis.__rescanLog) globalThis.__rescanLog.push({ t, x: b.x, y: b.y, w: b.w, h: b.h, s: b.s, rot: b.rot, dt });   // 動作確認用
          if (dt < BLURRED) continue;   // すでにぼかしが入っている
          if (isCleared(b, t, clearLog, W, H)) continue;   // 撮影中に「ぼかしなし」にした人
          faces.push(b);
        }
        samples.push({ t, faces });
        if (onProgress && dur) onProgress(Math.min(1, t / dur));
      }
      // 場面をまたいで同じ顔をつなぐ
      const tracks = [];
      for (const { t, faces } of samples) {
        for (const b of faces) {
          const c0 = center(b);
          let best = null, bd = Infinity;
          for (const tr of tracks) {
            const last = tr[tr.length - 1];
            if (t - last.t > LINK_GAP || last.t === t) continue;
            const q = center(last.b), d = Math.hypot(c0.x - q.x, c0.y - q.y);
            if (d < Math.max(b.w, last.b.w) * 0.8 && d < bd) { bd = d; best = tr; }
          }
          if (best) best.push({ t, b }); else tracks.push([{ t, b }]);
        }
      }
      const masks = [];
      for (const tr of tracks) {
        if (tr.length < 2 && tr[0].b.s < SURE) continue;   // 1回だけで点数も低いものは見送る
        const keys = tr.map(({ t, b }) => {
          const w = b.w * (b.rot === 90 || b.rot === 270 ? GROW_H : GROW_W), h = b.h * (b.rot === 90 || b.rot === 270 ? GROW_W : GROW_H);
          const q = center(b), up = b.rot === 0 ? b.h * 0.08 : 0;
          return { t, x: q.x - w / 2, y: q.y - up - h / 2, w, h };
        });
        masks.push({ from: Math.max(0, tr[0].t - STEP * 1.5), to: Math.min(dur, tr[tr.length - 1].t + STEP * 1.5), keys });
      }
      if (onProgress) onProgress(1);
      return masks;
    } finally {
      if (ownY && yunet && yunet.worker) yunet.worker.terminate();
      v.removeAttribute('src'); v.load();
      URL.revokeObjectURL(url);
    }
  })();
  return job;
}
