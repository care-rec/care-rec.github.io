// 確認画面の「手動ぼかし」。名札・ベッドネーム・自動で隠れなかった顔などを、指で囲んで隠す。
// 囲みは動画の中の位置（映像の画素）で持ち、場面ごとの位置（keys）を覚える。
// 2つの場面で位置がちがえば、その間はなめらかに動かす（人と一緒に動く名札に合わせるため）。
// 書き出しは、元の動画を最初から再生しながら、囲みを描いた画面を録画し直す（動画の長さと同じだけ時間がかかる）。
// 映像はこのスマホの中だけで処理し、どこにも送らない。
import { pickMimeType } from './recorder.js';
import { PatchTracker, seekFrame } from './track.js';
import { fixMp4Duration } from './mp4fix.js';

const MIN_SIDE = 16;          // これより小さい囲みは作らない（映像の画素）
const HANDLE_CSS = 26;        // 大きさを変えるつまみの大きさ（画面上の点）
const KEY_SNAP = 0.15;        // この秒数以内の場面の位置は、同じ場面として上書きする
const TRACK_STEP = 0.1;       // 自動で追いかけるとき、何秒ごとに位置を覚えるか
const TRACK_SURE = 0.7;       // 形の比べ方でこれ以上似ていれば「自信あり」
const TRACK_GROW = 1.7;       // 自信がない場面での、ぼかしの大きさ（倍）
const TRACK_SHRINK = 0.06;    // 自信が戻ったとき、0.1秒ごとに元の大きさへ戻す量
const TRACK_RATE = 0.5;       // 自動で追いかけるときの再生の速さ（ゆっくり再生しながら、映ったコマごとに追いかける）

// ---- 囲みの位置 ----
export function rectAt(m, t) {
  if (t < m.from - 0.02 || t > m.to + 0.02) return null;
  const k = m.keys;
  if (t <= k[0].t) return k[0];
  if (t >= k[k.length - 1].t) return k[k.length - 1];
  for (let i = 1; i < k.length; i++) {
    if (t <= k[i].t) {
      const a = k[i - 1], b = k[i], f = (t - a.t) / Math.max(1e-6, b.t - a.t);
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, w: a.w + (b.w - a.w) * f, h: a.h + (b.h - a.h) * f };
    }
  }
  return k[k.length - 1];
}

function setKey(m, t, r) {
  const i = m.keys.findIndex((k) => Math.abs(k.t - t) < KEY_SNAP);
  const k = { t, x: r.x, y: r.y, w: r.w, h: r.h };
  if (i >= 0) m.keys[i] = k; else m.keys.push(k);
  m.keys.sort((a, b) => a.t - b.t);
}

// ---- 隠す描き方 ----
// ぼかし：囲みをとても小さく縮めてから、なめらかに引き伸ばす（文字や顔の形が読めなくなる）
const tiny = document.createElement('canvas');
const tctx = tiny.getContext('2d');
const mid = document.createElement('canvas');
const mctx = mid.getContext('2d');
// shape が 'oval' のときは、だ円の形に隠す（顔に合う形。'rect' は名札などの四角いもの向け）
function ovalPath(ctx, r) {
  ctx.beginPath();
  ctx.ellipse(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2, 0, 0, Math.PI * 2);
}
export function hide(ctx, src, r, kind, W, H, shape = 'rect') {
  const x = Math.max(0, Math.floor(r.x)), y = Math.max(0, Math.floor(r.y));
  const w = Math.min(W, Math.ceil(r.x + r.w)) - x, h = Math.min(H, Math.ceil(r.y + r.h)) - y;
  if (w < 2 || h < 2) return;
  const oval = shape === 'oval';
  if (kind === 'black') {
    ctx.fillStyle = '#000';
    if (oval) { ovalPath(ctx, r); ctx.fill(); } else ctx.fillRect(x, y, w, h);
    return;
  }
  try {
    const tw = Math.max(2, Math.min(10, Math.round(w / 14))), th = Math.max(2, Math.round(tw * h / w));
    const mw = tw * 4, mh = th * 4;
    if (mid.width !== mw || mid.height !== mh) { mid.width = mw; mid.height = mh; }
    if (tiny.width !== tw || tiny.height !== th) { tiny.width = tw; tiny.height = th; }
    mctx.imageSmoothingEnabled = true; mctx.imageSmoothingQuality = 'high';
    mctx.drawImage(src, x, y, w, h, 0, 0, mw, mh);
    tctx.imageSmoothingEnabled = true; tctx.imageSmoothingQuality = 'high';
    tctx.drawImage(mid, 0, 0, mw, mh, 0, 0, tw, th);
    ctx.save();
    if (oval) { ovalPath(ctx, r); ctx.clip(); }
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tiny, 0, 0, tw, th, x, y, w, h);
    ctx.fillStyle = 'rgba(255,255,255,.12)';
    ctx.fillRect(x, y, w, h);
    ctx.restore();
  } catch (e) {
    ctx.fillStyle = '#888';
    if (oval) { ovalPath(ctx, r); ctx.fill(); } else ctx.fillRect(x, y, w, h);
  }
}

// ---- 囲みを作る・動かす画面 ----
export class RedactEditor {
  constructor({ video, canvas, onChange }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onChange = onChange || (() => {});
    this.masks = [];
    this.sel = null;
    this.nextId = 1;
    this.active = false;
    this.drag = null;
    this.duration = 0;
    this.ptrs = new Map();   // 画面に触れている指
    this.pinch = null;
    canvas.addEventListener('pointerdown', (e) => this._down(e));
    canvas.addEventListener('pointermove', (e) => this._move(e));
    for (const ev of ['pointerup', 'pointercancel']) canvas.addEventListener(ev, (e) => this._up(e));
    const redraw = () => { if (this.active) this.draw(); };
    for (const ev of ['seeked', 'loadeddata', 'pause']) video.addEventListener(ev, redraw);
  }

  get time() { return this.video.currentTime || 0; }

  open(duration) {
    this.duration = duration;
    this.active = true;
    this._watch();
    this.draw();
  }

  close() { this.active = false; }

  reset() { this.masks = []; this.sel = null; this.nextId = 1; }

  // 再生中は、コマごとに描き直す
  _watch() {
    const v = this.video;
    const step = () => {
      if (!this.active) return;
      if (!v.paused) this.draw();
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step); else requestAnimationFrame(step);
    };
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step); else requestAnimationFrame(step);
  }

  draw() {
    const v = this.video, c = this.canvas;
    const W = v.videoWidth, H = v.videoHeight;
    if (!W || !H) return;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const g = this.ctx;
    try { g.drawImage(v, 0, 0, W, H); } catch (e) { return; }
    const t = this.time;
    for (const m of this.masks) {
      const r = this._liveRect(m, t);
      if (r) hide(g, v, r, m.kind, W, H, m.shape);
    }
    // 目印（書き出しには入らない）
    const s = this._scale();
    g.save();
    for (const m of this.masks) {
      const r = this._liveRect(m, t);
      if (!r) continue;
      const on = m === this.sel;
      g.lineWidth = (on ? 3 : 2) / s;
      g.setLineDash(on ? [] : [8 / s, 6 / s]);
      const outline = () => { if (m.shape === 'oval') { ovalPath(g, r); g.stroke(); } else g.strokeRect(r.x, r.y, r.w, r.h); };
      g.strokeStyle = 'rgba(0,0,0,.6)'; outline();
      g.lineWidth = (on ? 2 : 1.5) / s;
      g.strokeStyle = on ? '#ffd400' : '#fff'; outline();
      g.setLineDash([]);
      // 番号
      g.font = `bold ${Math.round(15 / s)}px sans-serif`;
      g.fillStyle = on ? '#ffd400' : '#fff';
      g.strokeStyle = '#000'; g.lineWidth = 3 / s;
      g.strokeText(String(m.no), r.x + 5 / s, r.y + 18 / s); g.fillText(String(m.no), r.x + 5 / s, r.y + 18 / s);
      if (on && m.shape === 'oval') {
        // だ円は、大きさのつまみの位置が分かるよう、外枠をうすく見せる
        g.setLineDash([4 / s, 4 / s]); g.lineWidth = 1 / s; g.strokeStyle = 'rgba(255,212,0,.7)';
        g.strokeRect(r.x, r.y, r.w, r.h); g.setLineDash([]);
      }
      if (on) {
        const hr = HANDLE_CSS / 2 / s;
        g.beginPath(); g.arc(r.x + r.w, r.y + r.h, hr, 0, Math.PI * 2);
        g.fillStyle = '#ffd400'; g.fill(); g.lineWidth = 2 / s; g.strokeStyle = '#000'; g.stroke();
        // 右上の赤い×：押すとこの囲みを消す（画面の下までスクロールしなくても消せるように）
        if (!this.drag) {
          const p = this._delPos(r);
          g.beginPath(); g.arc(p.x, p.y, hr, 0, Math.PI * 2);
          g.fillStyle = '#d93025'; g.fill(); g.lineWidth = 2 / s; g.strokeStyle = '#fff'; g.stroke();
          const k = hr * 0.42;
          g.beginPath(); g.moveTo(p.x - k, p.y - k); g.lineTo(p.x + k, p.y + k); g.moveTo(p.x + k, p.y - k); g.lineTo(p.x - k, p.y + k);
          g.lineWidth = 3 / s; g.strokeStyle = '#fff'; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
        }
      }
    }
    if (this.drag && this.drag.mode === 'new') {
      const r = this.drag.rect;
      g.setLineDash([8 / s, 6 / s]); g.lineWidth = 2 / s; g.strokeStyle = '#ffd400';
      ovalPath(g, r); g.stroke();
    }
    g.restore();
  }

  // 消す×の位置（囲みの右上。画面の外に出るときは内側に寄せる）
  _delPos(r) {
    const hr = HANDLE_CSS / 2 / this._scale(), W = this.canvas.width, H = this.canvas.height;
    return { x: Math.max(hr, Math.min(W - hr, r.x + r.w)), y: Math.max(hr, Math.min(H - hr, r.y)) };
  }

  // ドラッグ中は、指で動かしている位置をそのまま見せる
  _liveRect(m, t) {
    if (this.drag && this.drag.m === m && this.drag.rect) return this.drag.rect;
    return rectAt(m, t);
  }

  // 画面上の1点が、映像の何画素にあたるか
  _scale() {
    const b = this.canvas.getBoundingClientRect();
    const W = this.canvas.width || 1, H = this.canvas.height || 1;
    return Math.min(b.width / W, b.height / H) || 1;
  }

  _pt(e) {
    const b = this.canvas.getBoundingClientRect();
    const W = this.canvas.width, H = this.canvas.height, s = this._scale();
    const ox = (b.width - W * s) / 2, oy = (b.height - H * s) / 2;
    return { x: Math.max(0, Math.min(W, (e.clientX - b.left - ox) / s)), y: Math.max(0, Math.min(H, (e.clientY - b.top - oy) / s)) };
  }

  // 2本の指でつまむ：選んでいる囲み（なければ2本の指の間にある囲み）を広げたり縮めたりし、指と一緒に動かす
  _pinchStart() {
    const [a, b] = [...this.ptrs.values()];
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, t = this.time;
    let m = this.sel && rectAt(this.sel, t) ? this.sel : null;
    if (!m) m = [...this.masks].reverse().find((k) => { const r = rectAt(k, t); return r && mid.x >= r.x && mid.x <= r.x + r.w && mid.y >= r.y && mid.y <= r.y + r.h; }) || null;
    this.drag = null;
    if (!m) { this.pinch = { m: null }; this.draw(); return; }
    const r0 = { ...rectAt(m, t) };
    this.sel = m;
    this.pinch = { m, r0, mid0: mid, d0: Math.max(8, Math.hypot(a.x - b.x, a.y - b.y)) };
    this.drag = { mode: 'pinch', m, rect: { ...r0 } };
    this.onChange();
    this.draw();
  }

  _pinchMove() {
    const P = this.pinch;
    if (!P || !P.m || P.done || this.ptrs.size < 2) return;
    const [a, b] = [...this.ptrs.values()];
    const W = this.canvas.width, H = this.canvas.height;
    const k = Math.max(0.1, Math.min(10, Math.hypot(a.x - b.x, a.y - b.y) / P.d0));
    const w = Math.max(MIN_SIDE, Math.min(W * 1.5, P.r0.w * k)), h = Math.max(MIN_SIDE, Math.min(H * 1.5, P.r0.h * k));
    const cx = P.r0.x + P.r0.w / 2 + (a.x + b.x) / 2 - P.mid0.x, cy = P.r0.y + P.r0.h / 2 + (a.y + b.y) / 2 - P.mid0.y;
    this.drag.rect = { x: Math.max(-w / 2, Math.min(W - w / 2, cx - w / 2)), y: Math.max(-h / 2, Math.min(H - h / 2, cy - h / 2)), w, h };
    this.draw();
  }

  _down(e) {
    if (!this.active || !this.canvas.width) return;
    e.preventDefault();
    this.video.pause();
    this.canvas.setPointerCapture?.(e.pointerId);
    this.ptrs.set(e.pointerId, this._pt(e));
    if (this.ptrs.size === 2 && !this.pinch) { this._pinchStart(); return; }
    if (this.ptrs.size > 1 || this.pinch) return;
    const p = this._pt(e), t = this.time, s = this._scale();
    // 選んでいる囲みのつまみ → 大きさを変える
    if (this.sel) {
      const r = rectAt(this.sel, t);
      // 右上の赤い× → この囲みを消す
      if (r) {
        const q = this._delPos(r);
        if (Math.hypot(p.x - q.x, p.y - q.y) < HANDLE_CSS / s) { this.ptrs.delete(e.pointerId); this.removeSel(); return; }
      }
      if (r && Math.hypot(p.x - (r.x + r.w), p.y - (r.y + r.h)) < HANDLE_CSS / s) {
        this.drag = { mode: 'size', m: this.sel, p, r0: { ...r }, rect: { ...r } };
        return;
      }
    }
    // 囲みの中 → 選んで動かす（あとから作ったものを優先）
    for (let i = this.masks.length - 1; i >= 0; i--) {
      const m = this.masks[i], r = rectAt(m, t);
      if (r && p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) {
        this.sel = m;
        this.drag = { mode: 'move', m, p, r0: { ...r }, rect: null, moved: false };
        this.onChange();
        this.draw();
        return;
      }
    }
    // 何もない所 → 新しく囲む
    this.drag = { mode: 'new', p, rect: { x: p.x, y: p.y, w: 0, h: 0 } };
  }

  _move(e) {
    if (this.ptrs.has(e.pointerId)) this.ptrs.set(e.pointerId, this._pt(e));
    if (this.pinch) { e.preventDefault(); this._pinchMove(); return; }
    const d = this.drag;
    if (!d) return;
    e.preventDefault();
    const p = this._pt(e), W = this.canvas.width, H = this.canvas.height;
    if (d.mode === 'new') {
      d.rect = { x: Math.min(p.x, d.p.x), y: Math.min(p.y, d.p.y), w: Math.abs(p.x - d.p.x), h: Math.abs(p.y - d.p.y) };
    } else if (d.mode === 'move') {
      const dx = p.x - d.p.x, dy = p.y - d.p.y;
      if (!d.moved && Math.hypot(dx, dy) * this._scale() < 6) return;
      d.moved = true;
      d.rect = { x: Math.max(-d.r0.w / 2, Math.min(W - d.r0.w / 2, d.r0.x + dx)), y: Math.max(-d.r0.h / 2, Math.min(H - d.r0.h / 2, d.r0.y + dy)), w: d.r0.w, h: d.r0.h };
    } else if (d.mode === 'size') {
      d.rect = { x: d.r0.x, y: d.r0.y, w: Math.max(MIN_SIDE, p.x - d.r0.x), h: Math.max(MIN_SIDE, p.y - d.r0.y) };
    }
    this.draw();
  }

  _up(e) {
    this.ptrs.delete(e.pointerId);
    if (this.pinch) {
      // 指を1本離したところで決める。残りの指は、全部離すまで何もしない
      const P = this.pinch, d = this.drag;
      if (P.m && d && d.rect && !P.done) { setKey(P.m, this.time, d.rect); this.onChange(); }
      P.done = true;
      this.drag = null;
      if (!this.ptrs.size) this.pinch = null;
      this.draw();
      return;
    }
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    const t = this.time;
    if (d.mode === 'new') {
      const r = d.rect;
      if (r.w >= MIN_SIDE && r.h >= MIN_SIDE) {
        const m = { id: this.nextId, no: this.nextId, kind: 'blur', shape: 'oval', from: 0, to: this.duration, keys: [{ t, ...r }] };
        this.nextId++;
        this.masks.push(m);
        this.sel = m;
      } else {
        this.sel = null;   // 何もない所を軽く触ったら、選ぶのをやめる
      }
    } else if (d.rect && (d.mode === 'size' || d.moved)) {
      setKey(d.m, t, d.rect);
    }
    this.onChange();
    this.draw();
  }

  // ＋ボタン：画面のまん中に、顔くらいの大きさのだ円を置く。指で動かし、2本の指で大きさを変える
  addOval() {
    const W = this.canvas.width, H = this.canvas.height;
    if (!W || !H) return;
    this.video.pause();
    const w = Math.round(Math.min(W, H) * 0.32), h = Math.round(w * 1.25);
    const m = { id: this.nextId, no: this.nextId, kind: 'blur', shape: 'oval', from: 0, to: this.duration, keys: [{ t: this.time, x: (W - w) / 2, y: (H - h) / 2, w, h }] };
    this.nextId++;
    this.masks.push(m);
    this.sel = m;
    this.onChange();
    this.draw();
  }

  // 撮影後の探し直しで見つけた顔を、だ円のぼかしとして加える（auto の印を付ける）
  addFound(list) {
    for (const f of list) {
      this.masks.push({ id: this.nextId, no: this.nextId, kind: 'blur', shape: 'oval', auto: true, from: f.from, to: f.to, keys: f.keys.map((k) => ({ ...k })) });
      this.nextId++;
    }
    this.sel = null;
    this.onChange();
    this.draw();
  }

  // ---- 自動で追いかける ----
  // 選んでいる囲みの「今の場面での見た目」を手がかりに、少しずつ場面を進めながら同じ見た目の場所を探し、
  // その位置を覚えていく。似た場所が見つからなくなったら（隠れた・裏返ったなど）、そこで止まる。
  // 戻り値 { cancel(), done: Promise<{ until, lost }> }
  trackSel({ onProgress } = {}) {
    const m = this.sel, v = this.video;
    let cancelled = false;
    const job = { cancel() { cancelled = true; }, done: null };
    job.done = (async () => {
      if (!m) return { until: 0, lost: false };
      v.pause();
      const W = v.videoWidth, H = v.videoHeight;
      const t0 = this.time, end = Math.min(m.to, this.duration || m.to);
      let r = { ...rectAt(m, t0) || m.keys[0] };
      setKey(m, t0, r);
      const tracker = new PatchTracker(v, r, t0);
      let until = t0, lost = false, lastKey = t0;
      // 追いかけに自信がない場面では、ぼかしを一時的に大きくして、少しずれても隠れるようにする。
      // 大きくするときはすぐ、元に戻すときはゆっくり（ぼかしが急に縮んで名札が見えないように）
      let grow = 1, unsure = false;
      // 1つの場面の結果を受け取る。見失ったら false
      const accept = (t, res) => {
        if (!res) { lost = true; return false; }
        r = { ...res.r };
        r.x = Math.max(-r.w / 2, Math.min(W - r.w / 2, r.x)); r.y = Math.max(-r.h / 2, Math.min(H - r.h / 2, r.y));
        until = t;
        if (res.byColor || res.score < TRACK_SURE) unsure = true;
        // 位置は0.1秒ごとに覚える（前に覚えていた、この場面までの位置は置き換える）
        if (t - lastKey >= TRACK_STEP - 0.005 || t >= end - 0.005) {
          m.keys = m.keys.filter((k) => k.t <= lastKey + 0.01 || k.t > t + 0.01);
          grow = unsure ? TRACK_GROW : Math.max(1, grow - TRACK_SHRINK);
          unsure = false;
          const gw = r.w * grow, gh = r.h * grow;
          m.keys.push({ t, x: r.x - (gw - r.w) / 2, y: r.y - (gh - r.h) / 2, w: gw, h: gh });   // 0.1秒ごとなので、近い場面の上書きはしない
          m.keys.sort((p, q) => p.t - q.t);
          lastKey = t;
        }
        if (onProgress) onProgress((until - t0) / Math.max(0.01, end - t0));
        return true;
      };
      if (v.requestVideoFrameCallback) {
        // ゆっくり再生しながら、実際に映ったコマごとに追いかける。
        // 場面を飛び飛びに移す方法は、iPhone では近くの区切りのコマにまとめられてしまい、追いつけないことがあるため
        const muted = v.muted, rate = v.playbackRate;
        v.muted = true;
        v.playbackRate = TRACK_RATE;
        try {
          await new Promise((done) => {
            let fin = false, lastT = t0;
            const finish = () => { if (!fin) { fin = true; clearInterval(wd); done(); } };
            let stuck = performance.now();
            const wd = setInterval(() => { if (cancelled || performance.now() - stuck > 4000) finish(); }, 300);
            const step = (_now, meta) => {
              if (fin) return;
              if (cancelled) return finish();
              const t = Math.min(end, meta && typeof meta.mediaTime === 'number' ? meta.mediaTime : v.currentTime);
              if (t > lastT + 0.02) {
                stuck = performance.now();
                lastT = t;
                if (!accept(t, tracker.update(t))) return finish();
              }
              if (t >= end - 0.02 || v.ended) return finish();
              v.requestVideoFrameCallback(step);
            };
            v.requestVideoFrameCallback(step);
            v.addEventListener('ended', finish, { once: true });
            v.play().catch(() => finish());
          });
        } finally {
          v.pause();
          v.playbackRate = rate;
          v.muted = muted;
        }
        // 止まった場面（見失った所、または最後）を見せる
        await seekFrame(v, until);
      } else {
        for (let t = t0 + TRACK_STEP; t <= end + 1e-3; t += TRACK_STEP) {
          if (cancelled) break;
          const tt = Math.min(t, end);
          await seekFrame(v, tt);
          if (!accept(tt, tracker.update(tt))) break;
          this.draw();
        }
      }
      this.onChange();
      this.draw();
      return { until, lost };
    })();
    return job;
  }

  // ---- 選んでいる囲みの操作 ----
  setShape(shape) { if (this.sel) { this.sel.shape = shape; this.onChange(); this.draw(); } }
  setKind(kind) { if (this.sel) { this.sel.kind = kind; this.onChange(); this.draw(); } }
  // 「今の場面から隠す」：終わりが今より前なら、終わりを動画の最後にする（隠す時間が0にならないように）
  setFrom() {
    const m = this.sel;
    if (!m) return;
    const t = this.time;
    m.from = t;
    if (m.to <= t + 0.05) m.to = this.duration;
    this.onChange(); this.draw();
  }
  // 「今の場面まで隠す」：始まりが今より後なら、始まりを動画の最初にする
  setTo() {
    const m = this.sel;
    if (!m) return;
    const t = this.time;
    m.to = t;
    if (m.from >= t - 0.05) m.from = 0;
    this.onChange(); this.draw();
  }
  setAll() { if (this.sel) { this.sel.from = 0; this.sel.to = this.duration; this.onChange(); this.draw(); } }
  removeSel() {
    if (!this.sel) return;
    this.masks = this.masks.filter((m) => m !== this.sel);
    this.sel = null;
    this.onChange();
    this.draw();
  }
  select(id) { this.sel = this.masks.find((m) => m.id === id) || null; this.onChange(); this.draw(); }
}

// ---- 書き出し ----
// 元の動画を再生しながら、囲みを隠した画面を録画し直す。
// 音声は、動画の音をスピーカーに出さずにそのまま録音へつなぐ。
// 必ず「押した」その場で呼ぶこと（iPhoneでは、押した直後でないと再生と音の準備が許されない）
export function exportRedacted(blob, masks, { durationSec, onProgress } = {}) {
  const url = URL.createObjectURL(blob);
  const v = document.createElement('video');
  v.playsInline = true;
  v.preload = 'auto';
  v.src = url;
  let ac = null, audioTracks = [];
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    ac = new AC();
    const src = ac.createMediaElementSource(v);
    const dest = ac.createMediaStreamDestination();
    src.connect(dest);
    audioTracks = dest.stream.getAudioTracks();
  } catch (e) {
    v.muted = true;   // 音を扱えない端末では、映像だけ書き出す
    audioTracks = [];
  }
  // 押した直後に一度再生して、あとで再生してよい状態にしておく
  const unlock = v.play().catch(() => {});
  if (ac && ac.state === 'suspended') ac.resume().catch(() => {});

  let cancelled = false, rec = null, finish = null;
  const job = {
    cancel() { cancelled = true; if (finish) finish(); },
    done: null,
  };
  job.done = (async () => {
    try {
      await unlock;
      if (!v.videoWidth) await new Promise((r) => v.addEventListener('loadedmetadata', r, { once: true }));
      v.pause();
      if (cancelled) throw new Error('cancelled');
      const W = v.videoWidth, H = v.videoHeight;
      // iPhone は録画の長さを短く読み違えることがあるので、録画した時間と比べて長いほうを使う
      const dur = Math.max(Number.isFinite(v.duration) && v.duration > 0 ? v.duration : 0, durationSec || 0);
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      const g = c.getContext('2d', { alpha: false });
      const paint = (t) => {
        g.drawImage(v, 0, 0, W, H);
        for (const m of masks) { const r = rectAt(m, t); if (r) hide(g, v, r, m.kind, W, H, m.shape); }
      };
      // 最初の場面へ戻して、最初のコマを描いてから録画を始める
      v.currentTime = 0;
      await new Promise((r) => { v.addEventListener('seeked', r, { once: true }); setTimeout(r, 1500); });
      paint(0);

      const mime = pickMimeType();
      if (mime === null || !c.captureStream) throw new Error('この端末では書き出しができません');
      const stream = c.captureStream(30);
      for (const t of audioTracks) stream.addTrack(t);
      const opts = { videoBitsPerSecond: 5_000_000 };
      if (mime) opts.mimeType = mime;
      const chunks = [];
      rec = new MediaRecorder(stream, opts);
      rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      const stopped = new Promise((r) => { rec.onstop = r; });
      rec.start(1000);

      // 再生しながら描く
      let reached = 0;
      await new Promise((done) => {
        let lastT = -1, stuckAt = performance.now();
        // 再生が止まってコマが来なくなっても、待ち続けないようにする
        const wd = setInterval(() => { if (performance.now() - stuckAt > 4000) resolve(); }, 1000);
        const resolve = () => { clearInterval(wd); done(); };
        finish = resolve;
        const step = (_now, meta) => {
          if (cancelled) return resolve();
          const t = meta && typeof meta.mediaTime === 'number' ? meta.mediaTime : v.currentTime;
          paint(t);
          if (onProgress && dur) onProgress(Math.min(1, t / dur));
          if (t !== lastT) { lastT = t; stuckAt = performance.now(); reached = Math.max(reached, t); }
          if (v.ended || (dur && t >= dur - 0.03)) return resolve();
          // 再生が止まったまま進まなければ、そこで終える
          if (performance.now() - stuckAt > 4000) return resolve();
          if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step); else requestAnimationFrame(() => step());
        };
        v.addEventListener('ended', () => resolve(), { once: true });
        v.play().then(() => {
          if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step); else requestAnimationFrame(() => step());
        }).catch(() => resolve());
      });
      v.pause();
      rec.stop();
      await stopped;
      if (cancelled) throw new Error('cancelled');
      // 最後まで読めずに途中で終わった動画は渡さない（短くなった動画を保存してしまわないため）
      if (durationSec && reached < durationSec - 1.5) throw new Error('short');
      const type = (rec.mimeType || chunks[0]?.type || 'video/mp4').split(';')[0];
      const out = await fixMp4Duration(new Blob(chunks, { type }));
      return { blob: out, type };
    } finally {
      try { if (rec && rec.state !== 'inactive') rec.stop(); } catch (e) { /* 何もしない */ }
      try { ac && ac.close(); } catch (e) { /* 何もしない */ }
      v.removeAttribute('src'); v.load();
      URL.revokeObjectURL(url);
    }
  })();
  return job;
}
