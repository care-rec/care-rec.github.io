// 手動ぼかしの「自動で追いかける」の中身。囲みの中の見た目（色つき）を手がかりに、次の場面で同じ見た目の場所を探す。
// - 色も比べる（赤い名札など、白黒では背景や肌と見分けにくいものも追える）
// - 囲みのまん中ほど重く見る（囲みが名札より大きく、背景が多く入っていても、背景につられにくい）
// - 前の動きから行き先を予想し、広めに探す（手で持って速く動かしても追いつく）
// - 形が合わないとき（回った・手で一部隠れた）は、その物にだけある色が集まっている所を追う
// - 見失ったら、無理に探さずに止まる（別の場所へ飛んで黙ってずれるのを防ぐ）
// 映像はこのスマホの中だけで処理し、どこにも送らない。

const SIZE = 48;          // 囲みの長い辺をこの大きさに縮めて比べる
const OK = 0.45;          // これ未満は「似ていない」
const SURE = 0.7;          // 形の比べ方でこれ以上似ていれば、そのまま信じる
const COLOR_OK = 0.55;     // 色の手がかりで「その物らしさ」が最初のこの割合以上あれば、そこにあるとみなす
const JUMP_OK = 0.7;       // 予想から大きく離れた所へ移るのに要る似かた
const LEARN = 0.72;       // これ以上よく似ていたら、手がかりを少し新しくする

const cv = document.createElement('canvas');
const g = cv.getContext('2d', { willReadFrequently: true });
// 映像の (x,y,w,h) を ow×oh に縮めて、RGB の並び（0〜255）で返す。映像の外は灰色
function grab(video, x, y, w, h, ow, oh) {
  if (cv.width !== ow || cv.height !== oh) { cv.width = ow; cv.height = oh; }
  g.fillStyle = '#808080';
  g.fillRect(0, 0, ow, oh);
  g.drawImage(video, x, y, w, h, 0, 0, ow, oh);
  const d = g.getImageData(0, 0, ow, oh).data, a = new Float32Array(ow * oh * 3);
  for (let i = 0, j = 0; i < d.length; i += 4) { a[j++] = d[i]; a[j++] = d[i + 1]; a[j++] = d[i + 2]; }
  return a;
}

// まん中ほど重い重み（だ円の外側は 0.25）
function weights(tw, th) {
  const w = new Float32Array(tw * th);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
    const dx = (x + 0.5 - tw / 2) / (tw / 2), dy = (y + 0.5 - th / 2) / (th / 2), d = dx * dx + dy * dy;
    w[y * tw + x] = d < 0.35 ? 1 : d < 1 ? 1 - 0.75 * (d - 0.35) / 0.65 : 0.25;
  }
  return w;
}

// 手がかり T（tw×th の RGB）を、重み W で平均を引いた形にしておく
function prep(T, W) {
  let sw = 0, s = 0;
  for (let i = 0; i < W.length; i++) { const k = W[i]; sw += k * 3; s += k * (T[i * 3] + T[i * 3 + 1] + T[i * 3 + 2]); }
  const m = s / sw, Tc = new Float32Array(T.length);
  let v = 0;
  for (let i = 0; i < W.length; i++) for (let c = 0; c < 3; c++) { const q = T[i * 3 + c] - m; Tc[i * 3 + c] = q; v += W[i] * q * q; }
  return { Tc, v, sw };
}

// 探す範囲 S（sw×sh）の中で、手がかりと最も似ている位置。range があれば、その範囲の位置だけ調べる
function search(S, sw, sh, P, W, tw, th, range) {
  const x0 = range ? Math.max(0, range.x0) : 0, x1 = range ? Math.min(sw - tw, range.x1) : sw - tw;
  const y0 = range ? Math.max(0, range.y0) : 0, y1 = range ? Math.min(sh - th, range.y1) : sh - th;
  let best = { x: 0, y: 0, score: -2 };
  if (P.v < 1e-6) return best;
  for (let oy = y0; oy <= y1; oy++) {
    for (let ox = x0; ox <= x1; ox++) {
      let s = 0, s2 = 0, st = 0;
      for (let j = 0; j < th; j++) {
        let p = ((oy + j) * sw + ox) * 3, q = j * tw;
        for (let i = 0; i < tw; i++, q++) {
          const k = W[q], a = S[p++], b = S[p++], c = S[p++], t = q * 3;
          s += k * (a + b + c); s2 += k * (a * a + b * b + c * c);
          st += k * (a * P.Tc[t] + b * P.Tc[t + 1] + c * P.Tc[t + 2]);
        }
      }
      const vs = s2 - (s * s) / P.sw;
      if (vs < 1e-3) continue;
      const sc = st / Math.sqrt(vs * P.v);
      if (sc > best.score) best = { x: ox, y: oy, score: sc };
    }
  }
  return best;
}

// 2分の1に縮めた絵（RGB）
function half(A, w, h) {
  const hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1), B = new Float32Array(hw * hh * 3);
  for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) for (let c = 0; c < 3; c++) {
    const i = ((y * 2) * w + x * 2) * 3 + c;
    B[(y * hw + x) * 3 + c] = (A[i] + A[i + 3] + A[i + w * 3] + A[i + w * 3 + 3]) / 4;
  }
  return { B, hw, hh };
}

// ---- 色の手がかり ----
// 色を 8×8×8 に分けた番号
const bin = (r, g, b) => ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
// 囲みの中にはよく出て、まわりには少ない色ほど 1 に近い「その物らしさ」の表を作る
function colorTable(inside, W, ring) {
  const q = new Float32Array(512), bg = new Float32Array(512);
  let qs = 0, bs = 0;
  for (let i = 0; i < W.length; i++) { const k = W[i]; q[bin(inside[i * 3], inside[i * 3 + 1], inside[i * 3 + 2])] += k; qs += k; }
  for (let i = 0; i < ring.length; i += 3) { bg[bin(ring[i], ring[i + 1], ring[i + 2])] += 1; bs += 1; }
  const p = new Float32Array(512);
  for (let c = 0; c < 512; c++) {
    const a = q[c] / Math.max(1, qs), b = bg[c] / Math.max(1, bs);
    p[c] = a > 0 ? a / (a + b * 1.5 + 1e-4) : 0;
  }
  return p;
}

export class PatchTracker {
  // video：映像。r：今の場面での囲み（映像の画素）。t：今の時刻
  constructor(video, r, t) {
    this.video = video;
    this.r = { ...r };
    this.t = t;
    this.vx = 0; this.vy = 0;
    this.sc = Math.min(1, SIZE / Math.max(8, r.w, r.h));
    this.tw = Math.max(8, Math.round(r.w * this.sc)) & ~1;
    this.th = Math.max(8, Math.round(r.h * this.sc)) & ~1;
    this.W = weights(this.tw, this.th);
    this.W2 = weights(this.tw >> 1, this.th >> 1);
    this.init = grab(video, r.x, r.y, r.w, r.h, this.tw, this.th);
    this.run = this.init.slice();
    this._setTemplate(this.init);
    // 色の手がかり：囲みの中と、まわり（囲みの3倍の範囲）の色を比べる
    const R = grab(video, r.x - r.w, r.y - r.h, r.w * 3, r.h * 3, this.tw * 3, this.th * 3);
    const ring = [];
    for (let y = 0; y < this.th * 3; y++) for (let x = 0; x < this.tw * 3; x++) {
      if (x >= this.tw && x < this.tw * 2 && y >= this.th && y < this.th * 2) continue;
      const i = (y * this.tw * 3 + x) * 3;
      ring.push(R[i], R[i + 1], R[i + 2]);
    }
    this.prob = colorTable(this.init, this.W, ring);
    this.probBase = this._probMean(this.init, this.tw, this.th, 0, 0) || 1e-3;
  }

  // 範囲 S の (ox,oy) から tw×th の窓の中の「その物らしさ」の平均
  _probMean(S, sw, sh, ox, oy) {
    let s = 0, n = 0;
    for (let y = Math.max(0, oy); y < Math.min(sh, oy + this.th); y++) for (let x = Math.max(0, ox); x < Math.min(sw, ox + this.tw); x++) {
      const i = (y * sw + x) * 3;
      s += this.prob[bin(S[i], S[i + 1], S[i + 2])]; n++;
    }
    return n ? s / n : 0;
  }

  // 色の手がかりで、その物らしい色が集まっている所へ窓を寄せていく（形が変わっても、回っても追える）
  _meanShift(S, sw, sh, ox, oy) {
    const tw = this.tw, th = this.th;
    for (let it = 0; it < 12; it++) {
      let sx = 0, sy = 0, sw8 = 0;
      for (let y = Math.max(0, oy); y < Math.min(sh, oy + th); y++) for (let x = Math.max(0, ox); x < Math.min(sw, ox + tw); x++) {
        const i = (y * sw + x) * 3, p = this.prob[bin(S[i], S[i + 1], S[i + 2])];
        if (p < 0.3) continue;
        sx += p * x; sy += p * y; sw8 += p;
      }
      if (sw8 < 1e-3) break;
      const nx = Math.round(sx / sw8 - tw / 2), ny = Math.round(sy / sw8 - th / 2);
      if (Math.abs(nx - ox) < 1 && Math.abs(ny - oy) < 1) break;
      ox = nx; oy = ny;
    }
    return { x: ox, y: oy, conf: this._probMean(S, sw, sh, ox, oy) / this.probBase };
  }

  _setTemplate(T) {
    this.P = prep(T, this.W);
    const h = half(T, this.tw, this.th);
    this.P2 = prep(h.B, this.W2);
  }

  // 映像がいま時刻 t の場面にあるとして、囲みの新しい位置を探す。{ r, score } か、見失えば null
  update(t) {
    const v = this.video;
    const dt = Math.max(0.01, t - this.t), r = this.r, sc = this.sc, tw = this.tw, th = this.th;
    // 前の動きから行き先を予想し、そのまわりを広めに探す
    const px = r.x + this.vx * dt, py = r.y + this.vy * dt;
    const mx = Math.max(r.w * 1.2, Math.abs(this.vx * dt) * 1.5 + r.w * 0.6);
    const my = Math.max(r.h * 1.2, Math.abs(this.vy * dt) * 1.5 + r.h * 0.6);
    const sx = px - mx, sy = py - my, sw = Math.round((r.w + mx * 2) * sc) & ~1, sh = Math.round((r.h + my * 2) * sc) & ~1;
    const S = grab(v, sx, sy, sw / sc, sh / sc, sw, sh);
    // 粗く（半分の大きさで）探してから、細かく合わせる
    const S2 = half(S, sw, sh);
    const c = search(S2.B, S2.hw, S2.hh, this.P2, this.W2, tw >> 1, th >> 1);
    let best = search(S, sw, sh, this.P, this.W, tw, th, { x0: c.x * 2 - 3, x1: c.x * 2 + 3, y0: c.y * 2 - 3, y1: c.y * 2 + 3 });
    let nx = sx + best.x / sc, ny = sy + best.y / sc;
    // 形の比べ方で自信がないときは、色の手がかりで探す（名札が回った・手で一部隠れた、など）
    let byColor = false;
    if (best.score < SURE) {
      // 予想した行き先と、形の比べ方で見つけた所の両方から寄せてみて、よりその物らしい方をとる
      let ms = this._meanShift(S, sw, sh, Math.round((px - sx) * sc), Math.round((py - sy) * sc));
      if (best.score >= OK) { const m2 = this._meanShift(S, sw, sh, best.x, best.y); if (m2.conf > ms.conf) ms = m2; }
      if (ms.conf >= COLOR_OK) {
        nx = sx + ms.x / sc; ny = sy + ms.y / sc;
        best = { score: Math.max(best.score, OK), color: ms.conf };
        byColor = true;
      }
    }
    // 予想した行き先から大きく離れた所は、よほどよく似ていなければ信じない（背景などへ飛んで、黙ってずれるのを防ぐ）
    if (globalThis.__trackLog) globalThis.__trackLog.push({ t: +t.toFixed(2), s: +best.score.toFixed(3), c: best.color ? +best.color.toFixed(2) : undefined, d: +(Math.min(Math.hypot(nx - px, ny - py), Math.hypot(nx - r.x, ny - r.y)) / Math.max(r.w, r.h)).toFixed(2) });   // 動作確認用
    if (!byColor && best.score >= OK && Math.min(Math.hypot(nx - px, ny - py), Math.hypot(nx - r.x, ny - r.y)) > Math.max(r.w, r.h) * 0.6 && best.score < JUMP_OK) best.score = 0;
    // 似た場所が見つからなければ、見失ったとして止まる（画面全体から探し直すと、模様の少ない所では
    // 別の場所へ飛んでしまい、黙ってずれるほうが危ないため）
    if (best.score < OK) return null;
    this.vx = this.vx * 0.4 + ((nx - r.x) / dt) * 0.6;
    this.vy = this.vy * 0.4 + ((ny - r.y) / dt) * 0.6;
    this.r = { x: nx, y: ny, w: r.w, h: r.h };
    this.t = t;
    // よく似ていたら、見た目の変化に合わせて手がかりを少し新しくする（最初の見た目も半分残す）
    if (best.score >= LEARN && !byColor) {
      const now = grab(v, nx, ny, r.w, r.h, tw, th);
      for (let i = 0; i < now.length; i++) this.run[i] = this.run[i] * 0.85 + now[i] * 0.15;
      const T = new Float32Array(now.length);
      for (let i = 0; i < T.length; i++) T[i] = (this.init[i] + this.run[i]) / 2;
      this._setTemplate(T);
    }
    return { r: this.r, score: best.score, byColor };
  }
}

// 動画を時刻 t の場面へ移し、その場面の絵が本当に描けるようになるまで待つ。
// iPhone の Safari では「移り終わった」の知らせのあとも、少しの間は前の場面の絵が残ることがあるため、
// 新しい絵が出たという知らせ（requestVideoFrameCallback）も待つ
export function seekFrame(v, t) {
  return new Promise((ok) => {
    let fin = false;
    const end = () => { if (!fin) { fin = true; clearTimeout(to); v.removeEventListener('seeked', onSeeked); ok(); } };
    const to = setTimeout(end, 3000);
    const onSeeked = () => {
      v.removeEventListener('seeked', onSeeked);
      if (v.requestVideoFrameCallback) {
        v.requestVideoFrameCallback(() => end());
        setTimeout(end, 120);   // 知らせが来ない端末でも止まらない
      } else setTimeout(end, 60);
    };
    v.addEventListener('seeked', onSeeked);
    v.currentTime = t;
  });
}
