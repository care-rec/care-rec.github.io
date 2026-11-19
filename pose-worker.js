// 体の姿勢から頭の位置を割り出す（遠くの小さな顔は、顔そのものより体の形のほうが見つけやすい）。
// 画面の動きを止めないよう、別の流れ（ワーカー）で動かす。映像はこのスマホの中だけで処理し、どこにも送らない。
import { FilesetResolver, PoseLandmarker } from './vendor/tasks-vision/vision_bundle.mjs';

// 部品の読み込みに使う importScripts は、この種類のワーカーでは使えないので、同じ働きをするものに置きかえる
self.importScripts = (...urls) => {
  for (const url of urls) {
    const x = new XMLHttpRequest();
    x.open('GET', url, false);
    x.send();
    if (x.status && x.status !== 200) throw new Error('load failed: ' + url);
    (0, eval)(x.responseText);
  }
};

let pose = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const vision = await FilesetResolver.forVisionTasks(m.wasm);
      pose = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: m.model, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numPoses: 2,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
      });
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'detect') {
      const res = pose.detect(m.bitmap);
      m.bitmap.close();
      const people = res.landmarks.map((lm) => lm.map((p) => ({ x: p.x, y: p.y, v: p.visibility ?? 0 })));
      self.postMessage({ type: 'result', id: m.id, people });
    }
  } catch (err) {
    if (m.bitmap) try { m.bitmap.close(); } catch (_) { /* なし */ }
    self.postMessage({ type: m.type === 'init' ? 'fail' : 'result', id: m.id, people: [], error: String(err && err.message || err) });
  }
};
