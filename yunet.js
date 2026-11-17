// YuNet（もう1つの顔検出）をワーカーで動かすための窓口。
// 使えない環境（古いブラウザなど）では読み込みに失敗するだけで、アプリはこれまでの検出だけで動く。

export class YuNetDetector {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.nextId = 1;
    this.waiting = new Map();
  }

  load(timeoutMs = 30000) {
    if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap === 'undefined') {
      return Promise.resolve(false);
    }
    return new Promise((resolve) => {
      let done = false;
      const finish = (ok) => { if (!done) { done = true; resolve(ok); } };
      try {
        this.worker = new Worker(new URL('./yunet-worker.js', import.meta.url), { type: 'module' });
      } catch (e) {
        finish(false);
        return;
      }
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { this.ready = true; finish(true); }
        else if (m.type === 'fail') finish(false);
        else if (m.type === 'result') {
          const w = this.waiting.get(m.id);
          if (w) { this.waiting.delete(m.id); w(m.boxes || []); }
        }
      };
      this.worker.onerror = () => finish(false);
      this.worker.postMessage({ type: 'init', model: new URL('./vendor/yunet/face_detection_yunet_2023mar.onnx', import.meta.url).href });
      setTimeout(() => finish(false), timeoutMs);
    });
  }

  // source（canvas）のいまの絵を写し取って、区画（cells）ごとに顔を探す。結果は元の座標の枠の一覧
  // cells: [{ cx, cy, side, rot, ox, oy, size }]（区画の位置・大きさは 640x640 の入力画像の中での値）
  async detect(source, cells) {
    if (!this.ready) return [];
    const bitmap = await createImageBitmap(source);
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.waiting.set(id, resolve);
      this.worker.postMessage({ type: 'detect', id, bitmap, cells }, [bitmap]);
    });
  }

  // 画面全体を rot 度回して1区画で調べるときの区画
  static fullCell(W, H, rot) {
    const a = rot * Math.PI / 180, c = Math.abs(Math.cos(a)), s = Math.abs(Math.sin(a));
    return { cx: W / 2, cy: H / 2, side: Math.max(W * c + H * s, W * s + H * c), rot, ox: 0, oy: 0, size: 640 };
  }
}
