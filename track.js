// 手動ぼかしの「自動で追いかける」の中身。囲みの中の見た目（色つき）を手がかりに、次の場面で同じ見た目の場所を探す。
// - 色も比べる（赤い名札など、白黒では背景や肌と見分けにくいものも追える）
// - 囲みのまん中ほど重く見る（囲みが名札より大きく、背景が多く入っていても、背景につられにくい）
// - 前の動きから行き先を予想し、広めに探す（手で持って速く動かしても追いつく）
// - 見失ったら、無理に探さずに止まる（別の場所へ飛んで黙ってずれるのを防ぐ）
// 映像はこのスマホの中だけで処理し、どこにも送らない。

const SIZE = 48;          // 囲みの長い辺をこの大きさに縮めて比べる
const OK = 0.45;          // これ未満は「似ていない」
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
    const nx = sx + best.x / sc, ny = sy + best.y / sc;
    // 予想した行き先から大きく離れた所は、よほどよく似ていなければ信じない（背景などへ飛んで、黙ってずれるのを防ぐ）
    if (best.score >= OK && Math.min(Math.hypot(nx - px, ny - py), Math.hypot(nx - r.x, ny - r.y)) > Math.max(r.w, r.h) * 0.6 && best.score < JUMP_OK) best.score = 0;
    // 似た場所が見つからなければ、見失ったとして止まる（画面全体から探し直すと、模様の少ない所では
    // 別の場所へ飛んでしまい、黙ってずれるほうが危ないため）
    if (best.score < OK) return null;
    this.vx = this.vx * 0.4 + ((nx - r.x) / dt) * 0.6;
    this.vy = this.vy * 0.4 + ((ny - r.y) / dt) * 0.6;
    this.r = { x: nx, y: ny, w: r.w, h: r.h };
    this.t = t;
    // よく似ていたら、見た目の変化に合わせて手がかりを少し新しくする（最初の見た目も半分残す）
    if (best.score >= LEARN) {
      const now = grab(v, nx, ny, r.w, r.h, tw, th);
      for (let i = 0; i < now.length; i++) this.run[i] = this.run[i] * 0.85 + now[i] * 0.15;
      const T = new Float32Array(now.length);
      for (let i = 0; i < T.length; i++) T[i] = (this.init[i] + this.run[i]) / 2;
      this._setTemplate(T);
    }
    return { r: this.r, score: best.score };
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
