// 姿勢から頭の位置を割り出す部品の窓口。
// 使えない環境では読み込みに失敗するだけで、アプリはこれまでの検出だけで動く。

export class PoseFinder {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.nextId = 1;
    this.waiting = new Map();
  }

  load(timeoutMs = 30000) {
    if (typeof Worker === 'undefined' || typeof createImageBitmap === 'undefined') return Promise.resolve(false);
    return new Promise((resolve) => {
      let done = false;
      const finish = (ok) => { if (!done) { done = true; resolve(ok); } };
      try {
        this.worker = new Worker(new URL('./pose-worker.js', import.meta.url), { type: 'module' });
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
          if (w) { this.waiting.delete(m.id); w(m.people || []); }
        }
      };
      this.worker.onerror = () => finish(false);
      this.worker.postMessage({
        type: 'init',
        wasm: new URL('./vendor/tasks-vision/wasm', import.meta.url).href,
        model: new URL('./vendor/pose/pose_landmarker_lite.task', import.meta.url).href,
      });
      setTimeout(() => finish(false), timeoutMs);
    });
  }

  // source（canvas）のいまの絵から、写っている人ごとの体の点（0〜1の座標）を返す。
  // crop を渡すと、その範囲だけを拡大して調べる（座標はその範囲の中での0〜1）
  async detect(source, crop) {
    if (!this.ready) return [];
    const bitmap = crop ? await createImageBitmap(source, crop.x, crop.y, crop.w, crop.h) : await createImageBitmap(source);
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.waiting.set(id, resolve);
      this.worker.postMessage({ type: 'detect', id, bitmap }, [bitmap]);
    });
  }
}
