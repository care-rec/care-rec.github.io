// カメラ取得・顔検出・ぼかし描画・タップ切り替え・追跡
//
// 安全の考え方：
//  - カメラ映像は「raw」という内部の画面にだけ描く（録画されない）
//  - 録画されるのは「out」だけ。outには必ず、ぼかしをかけた後の絵しか描かない
//  - 顔検出の準備ができていない間は、画面全体をぼかす
//  - 顔を見失っても、しばらくは直前の位置をぼかし続ける

import { FilesetResolver, FaceDetector } from './vendor/tasks-vision/vision_bundle.mjs';
import { loadFaceId, faceIdReady, cropAligned, describe, distance } from './face-id.js';
import { YuNetDetector } from './yunet.js';
import { PoseFinder } from './pose.js';

const HOLD_BLUR_MS = 1200;   // ぼかす顔を見失ってから、ぼかしを残す時間
const HOLD_CLEAR_MS = 800;   // ぼかさない顔を見失ってから、選択を覚えておく時間（過ぎると選択解除＝ぼかしに戻る）
const HOLD_EDGE_MS = 3500;   // 画面の端で見失った顔（半分はみ出した顔は検出しにくい）は、長めにぼかしを残す
const EXPAND_W = 1.9;        // ぼかし範囲の広げ方（顔の幅に対して）。ふちをやわらかくする分、広めにとる
const EXPAND_H = 2.2;
// 顔かどうかの調べ直し（ライトや家具を顔とまちがえた枠を消すため）
const HOLD_UNSURE_MS = 550;  // まだ顔と確かめられていない枠を見失ったとき、ぼかしを残す時間
// 判断は1回の点数ではなく、くり返し調べた平均で行う（ライトは 0.51〜0.68 を行き来し、本物の顔は 0.75〜0.94 で安定）
const VERIFY_OK = 0.72;      // 平均がこれ以上なら「顔」
const VERIFY_NG = 0.67;      // 3回以上調べて平均がこれ未満なら「顔ではない」として、ぼかしを消す
const VERIFY_STRONG = 0.8;   // 平均が一度でもこれ以上になった枠は、その後どう見えても「顔ではない」にはしない（横を向いた顔を守る）
const VERIFY_MIN_SAMPLES = 3;
const GONE_SCORE = 0.3;        // 調べ直しの点数がこれ未満＝顔らしさがまったく無い
const GONE_TIMES = 3;          // それがこの回数続いたら、「ずっと顔として扱う」保護も外して、ぼかしを消す（顔が去ったあとの背景に残るぼかし対策）
const BIRTH_OK = 0.5;         // 新しい枠をその場で調べた点数がこれ未満なら、最初からぼかさない
const BIRTH_OK_Y = 0.6;       // もう1つの顔検出が動いているときの基準（マスクの顔はそちらで見つかるので、ここは厳しくする）
const BIRTH_RETRIES = 5;      // ぼかさなかった枠を、短い間隔で調べ直す回数
// 回転させた向き（横倒し・逆さ）では基準を上げる。実測：寝ている顔は 0.75〜0.89（中央0.82）、
// ライトは回転させると 0.72 を超えることがあるが 0.8 には届かない
const VERIFY_OK_TURNED = 0.76;
const VERIFY_NG_TURNED = 0.72;
const VERIFY_UNSURE_MAX = 6;  // 判断がつかない状態がこの回数続いたら「顔ではない」とする（中間の点数をうろうろするライト対策）
const VERIFY_EVERY_MS = 150; // 判断がつくまでの調べ直しの間隔
const VERIFY_KEEP_MS = 500;  // 判断がついたあとも調べ続ける間隔
const MOVE_DIFF = 18;         // 縮小画面の明るさの変化（1点あたりの平均）がこれ以上なら「カメラを動かしている」
const MOVE_KEEP_MS = 600;     // 動きが止まってからも、この時間は「動かしている」扱いにする
const PROV_MS = 450;          // 仮にぼかした枠は、この時間内にもう1つの検出が顔と確かめなければ外す
const PROV_MAX_W = 0.4;       // 仮にぼかす候補の大きさの上限（画面の幅に対する割合）
const PROV_MIN = 0.55;        // カメラを動かしている間は、この点数以上の候補を、調べる前に仮にぼかす
const STRONG_SCORE = 0.75;   // 「顔ではない」とした枠でも、検出の点数がこれ以上になったらすぐぼかしに戻す
const FAKE_ZONE_MS = 8000;   // 「顔ではない」と分かった場所を覚えておく時間。同じ場所に出た枠は、ぼかす前に調べる
// もう1つの顔検出（YuNet）。マスク＋横顔・寝ている顔に強い。別の流れで、向きを変えながらくり返し探す
const Y_OK = 0.6;             // まっすぐの向きでこの点数以上なら「顔」
const Y_OK_TURNED = 0.7;      // 回転させた向きでは基準を上げる（ライトなどが通りにくいように）
const Y_AGREE = 0.55;          // もう1つの検出（BlazeFace）も同じ場所に反応しているときは、この点数で「顔」とする
const Y_SURE = 0.8;            // これ以上なら1回で「顔」とする。これ未満は、同じ場所で2回反応してはじめて「顔」とする
const Y_SURE_UP = 0.72;        // まっすぐの向きでの同じ基準
const Y_TWICE_MS = 800;        // 2回目の反応を待つ時間
const Y_FIRM = 0.75;           // 一度でもこの点数が出た枠は、本物の顔らしいので長めに追いかける
const FOLLOW_WEAK_MS = 1200;   // 点数の低い枠を、見た目だけで追いかける最大時間
const HOLD_WEAK_MS = 500;      // 点数の低い枠を、見失ったあと残す時間
const HOLD_SWING_MS = 300;     // カメラを動かしている間に見失った枠を残す時間（位置が合わなくなるので長く残さない）
const Y_PROP = 0.5;            // 画面全体の検査でこの点数以上なら「候補」として、その場所を拡大して調べ直す
const Y_PROP_OK = 0.7;         // 拡大して調べ直したときの合格点
const Y_PROP_OK_UP = 0.62;     // まっすぐの向きでの合格点
const Y_PROP_LIFE_MS = 600;    // 候補を調べ直すまでの期限
const Y_PROP_NG_MS = 1000;     // 調べ直して顔でなかった場所は、この時間は候補にしない
const Y_HINT = 0.45;          // これ以上なら「顔がまだそこにある」手がかりとして扱う
const Y_RECENT_MS = 3000;     // YuNetが顔と判断してからこの時間は、ほかの調べ直しで「顔ではない」にしない
const Y_GONE_MS = 2000;       // 「顔がもう無い」と判断するには、この時間YuNetの手がかりも無いこと
// 見た目の追いかけ（検出が途切れても、顔の見た目を手がかりに毎コマぼかしを動かす）
const GRAY_W = 160;           // 追いかけに使う縮小画面の幅
const NCC_OK = 0.5;           // 見た目の似ている度合い（-1〜1）がこれ以上なら、同じ顔とみなして追う
const FOLLOW_MAX_MS = 3000;   // 検出がなくても、見た目の追いかけだけでぼかしを続ける最大時間
// 処理量の調整（本体が熱くならないように）。カメラを振っている間だけ全力で探し、落ち着いているときは回数を減らす
const BF_EVERY_CALM = 2;      // 落ち着いているときは、もとの顔検出を何コマに1回行うか
const BF_EXTRA_EVERY = 4;     // 画面の一部を拡大して探す検出を、何コマに1回行うか
const VERIFY_EVERY_FRAMES = 3; // 顔かどうかの調べ直しを、何コマに1回行うか
const GM_RANGE = 7;           // カメラの動きを探す範囲（粗い画面の点の数。1点は元の画面の18ピクセルほど）
const GM_MIN_TEXTURE = 1.5;   // 映像の模様がこれより少ないときは、動きを測らない
const GM_MAX_RATIO = 0.8;     // いちばん合う位置が、平均よりこの割合以下でないときは、動きを測らない
const Y_IDLE_GAP_MS = 150;    // 顔がまだ見つかっていないときの休み時間（短めにして、早く見つける）
const Y_NEW_VIEW = 0.12;      // 前回の検査からカメラがこの割合（画面の幅に対して）動いたら、新しい景色として休まず調べる
const Y_CALM_GAP_MS = 250;    // 落ち着いているときの、もう1つの顔検出の休み時間
const Y_CALM_DUTY = 3;        // 休み時間は、1回の検出にかかった時間のこの倍以上とる（遅い端末ほど長く休む）
const Y_HOT_MS = 1200;         // カメラの動きが止まってから、全力で探し続ける時間
const Y_START_HOT_MS = 1500;  // カメラを開いた直後に全力で探す時間
const Y_TWIN_MAX_MS = 4000;   // 2つ同時に動かすのは、続けてこの時間まで（振り続けても発熱しすぎないように）
const EDGE_REACH = 0.5;       // ぼかしの端から画面の端までが、ぼかしの大きさのこの割合より近ければ、画面の端まで広げる
const EDGE_HOLD_MS = 2000;    // 画面の端からはみ出しかけた顔を、最後に検出されてからその位置でぼかし続ける長さ（上限）
const FLIP_COVER_MS = 1200;   // カメラを切り替えた直後、画面全体をぼかしておく時間
// 姿勢から頭の位置を割り出す（遠くの小さな顔は、顔の検出では見つけにくい）
const POSE_GAP_SWING_MS = 150; // カメラを振っている間の間隔
const POSE_GAP_MS = 350;      // 調べる間隔
const POSE_GAP_BUSY_MS = 1000; // 近くの顔をはっきり追えているときの間隔（発熱をおさえる）
const POSE_BIG_FACE = 0.14;   // 顔の幅が画面の幅のこの割合以上なら「近くの顔」
const POSE_DUTY = 3;          // 休み時間は、1回にかかった時間のこの倍以上とる
const POSE_VIS = 0.6;         // 体の点が「見えている」とみなす確かさ
const POSE_TWICE_MS = 1200;   // 同じ場所を2回指したら頭と認める。2回目を待つ時間
const POSE_PERSON_MS = 5000;  // 見つけた人の周りを拡大して調べ続ける時間
const POSE_ZOOM = 1.5;        // 人の大きさに対する、拡大して調べる範囲の広さ
const POSE_KEEP_MS = 2500;    // 頭の位置が分かってから、ぼかしを保つ時間
const POSE_BOX = 1.5;         // 頭の半径に対する、顔の枠の大きさ
// 映像を少し遅らせて画面に出す。検出は最新のコマで行うので、顔を見つけたら、まだ出していないコマにさかのぼってぼかせる
const OUT_DELAY_MS = 450;     // 遅らせる時間
const OUT_MAX_FRAMES = 18;    // ためておくコマ数の上限
const BACK_GROW = 1.12;       // さかのぼってぼかすときは、位置の誤差を見込んで少し広げる
const BLUR_STEPS = [2, 3, 4]; // にじませる幅（顔の範囲を72に縮めた画面での幅）。大きいほど強いぼかし
const BLUR_HAZE = 0.12;       // 白っぽさ（0〜1）
// ぼかしの動きをなめらかにする（脈打つような伸び縮み・小刻みなぶれを抑える）。
// 位置はならして動かし、大きさは「顔を覆うのに足りなければすぐ広げ、余ったらゆっくり縮める」。
// ならした位置から元の範囲がはみ出す分は必ず広げて覆うので、見逃しは増えない
const SM_POS_MS = 170;        // 位置をならす強さ（ゆっくり動くとき）
const SM_SNAP = 0.35;         // 顔の幅のこの割合以上ずれたら、ならさずにすぐ追いつく
const SM_SHRINK_MS = 1200;     // 縮むときの速さ
const SM_RESET_MS = 400;      // この時間ぼかしが途切れたら、ならし直す

// 顔の見分け（距離が小さいほど似ている。テストでは同じ人 0.33〜0.52、別人 0.69以上）
const ID_MATCH = 0.47;       // これ未満が2回続いたら「選んだ人」とみなしてぼかしを外す
const ID_MATCH_TIMES = 2;
const ID_MISMATCH = 0.68;    // ぼかしなしの顔が、これ以上の距離を3回続けたら別人とみなしてぼかしに戻す
const ID_MISMATCH_TIMES = 3;
const ID_LEARN_MIN = 0.25;   // 覚えている顔とこの範囲で違うとき、新しい角度として追加で覚える
const ID_MAX_SAMPLES = 15;
const ID_CHECK_BLUR_MS = 300;   // ぼかし中の顔を照合する間隔
const ID_CHECK_CLEAR_MS = 600;  // ぼかしなしの顔を確かめる間隔

export class BlurCamera {
  constructor({ video, out, overlay, onStatus }) {
    this.video = video;
    this.out = out;
    this.octx = out.getContext('2d', { alpha: false });
    this.overlay = overlay;
    this.vctx = overlay.getContext('2d');
    this.raw = document.createElement('canvas');
    this.rctx = this.raw.getContext('2d', { alpha: false, willReadFrequently: false });
    this.tile = document.createElement('canvas');
    this.tctx = this.tile.getContext('2d', { alpha: false });
    this.small = document.createElement('canvas');
    this.sctx = this.small.getContext('2d', { alpha: false });
    this.pre = document.createElement('canvas');
    this.pctx = this.pre.getContext('2d', { alpha: false });
    this.midA = document.createElement('canvas');
    this.actx = this.midA.getContext('2d', { alpha: false });
    this.midB = document.createElement('canvas');
    this.bctx = this.midB.getContext('2d');
    this.onToggle = null;
    this.onBirth = null;      // 動作確認用
    this.persons = [];        // ぼかさないと選んだ人の顔の特徴（メモリの中だけ。カメラを閉じると消える）
    this.nextPerson = 1;
    this.idBusy = false;
    this.idCanvas = document.createElement('canvas');
    this.fakeZones = [];
    this.vCanvas = document.createElement('canvas');
    this.vCanvas.width = 256; this.vCanvas.height = 256;
    this.vctx2 = this.vCanvas.getContext('2d', { alpha: false });
    this.onStatus = onStatus || (() => {});
    this.yunet = null;          // もう1つの顔検出（読み込めたときだけ使う）
    this.pose = null;           // 姿勢から頭を割り出す部品（読み込めたときだけ使う）
    this.poseBusy = false;
    this.pQueue = [];
    this.poseCands = [];
    this.delayQ = [];           // 遅らせて出すコマの列（メモリの中だけ。録画はされない）
    this.framePool = [];        // 使い回す画面
    this.shown = null;          // いま画面に出ているコマの情報
    this.yunet2 = null;         // 2つ目（同時に動かして、見つけるまでの時間を縮める）
    this.yBusy = false;
    this.yBusy2 = false;
    this.yQueue = [];
    this.yIdx = 0;
    this.grayCanvas = document.createElement('canvas');
    this.gctx = this.grayCanvas.getContext('2d', { willReadFrequently: true });
    this.gray = null;

    this.detector = null;
    this.detectorFailed = false;
    this.tracks = [];
    this.nextId = 1;
    this.facing = 'environment';
    this.stream = null;
    this.running = false;
    this.locked = false;      // 録画中は画面サイズを固定
    this.label = '';
    this.tileIndex = 0;
    this.slotIndex = 0;
    this.searchTurned = true;   // 寝ている人など、横倒し・逆さの顔も探すか
    this.lastTs = 0;
    this._loop = this._loop.bind(this);

    out.addEventListener('pointerdown', (e) => this._onTap(e));
  }

  get ready() { return !!this.detector && this.running && this.out.width > 0; }

  async initDetector() {
    if (this.detector) return;
    const vision = await FilesetResolver.forVisionTasks('./vendor/tasks-vision/wasm');
    const make = (delegate) => FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: './vendor/blaze_face_short_range.tflite', delegate },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.3,
      minSuppressionThreshold: 0.3,
    });
    try {
      this.detector = await make('GPU');
    } catch (e) {
      this.detector = await make('CPU');
    }
  }

  async start() {
    await this._openCamera();
    this.running = true;
    this._watchFrames();
    this.yStartAt = 0;
    requestAnimationFrame(this._loop);
    try {
      await this.initDetector();
    } catch (e) {
      this.detectorFailed = true;
      throw e;
    }
    // 顔を見分ける部品と、もう1つの顔検出も準備中に読み込む（読み込み中は映像が一瞬止まるため、撮影前に済ませる）。
    // 失敗しても、時間がかかっても、撮影はできる（遅れて読み込めたら、その時から使う）
    if (!this.yLoad) {
      const y = new YuNetDetector();
      this.yLoad = y.load().then((ok) => {
        if (!ok) return;
        this.yunet = y;
        // 姿勢の部品は、撮影の準備を待たせずに後ろで読み込む
        const p = new PoseFinder();
        p.load().then((ok2) => { if (ok2) this.pose = p; }).catch(() => {});
      }).catch(() => {});
    }
    await Promise.race([
      Promise.all([loadFaceId().catch(() => {}), this.yLoad]),
      new Promise((r) => setTimeout(r, 25000)),
    ]);
  }

  // カメラから新しいコマが届いたことを知る（同じ絵を何度も調べないため）
  _watchFrames() {
    const v = this.video;
    if (!v.requestVideoFrameCallback) return;
    const id = (this.frameWatch = (this.frameWatch || 0) + 1);
    const cb = () => {
      if (id !== this.frameWatch || !this.running) return;
      this.newFrame = true;
      this.frameSeenAt = performance.now();
      v.requestVideoFrameCallback(cb);
    };
    v.requestVideoFrameCallback(cb);
  }

  async _openCamera() {
    this._closeCamera();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: this.facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this._flushDelay();
    this.tracks = [];
    // 前のカメラで調べていた結果は捨てる
    this.yQueue = [];
    this.pQueue = [];
    this.poseCands = [];
    this.personBox = null;
    this.yRejects = [];
    this.camX = 0; this.camY = 0;
    this.grayPrev = null;
    this.camGen = (this.camGen || 0) + 1;   // 調べている途中の結果が、切り替え後に届いても使わない
  }

  _closeCamera() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  // 録画中でも切り替えられる。切り替えた直後は顔の位置がまだ分からないので、
  // 新しいカメラのコマが届き始めてから FLIP_COVER_MS の間は、画面全体をぼかす
  async flip() {
    if (this.flipping) return;
    this.flipping = true;
    const before = this.facing;
    this.facing = before === 'environment' ? 'user' : 'environment';
    this.coverFrom = -1;   // 新しいコマが届いたら、そこから数える
    try {
      await this._openCamera();
    } catch (e) {
      // 切り替えられなければ、元のカメラに戻す
      this.facing = before;
      try { await this._openCamera(); } catch (e2) { /* 何もしない */ }
      throw e;
    } finally {
      this.flipping = false;
    }
  }

  stop() {
    this.running = false;
    this._closeCamera();
    this._flushDelay();
    for (const c of this.framePool) { c.canvas.width = 0; c.canvas.height = 0; }
    this.framePool = [];
    this.tracks = [];
    this.persons = [];
  }

  lock(v) { this.locked = v; }
  setLabel(text) { this.label = text; }

  // ---- 毎フレームの処理 ----
  _loop() {
    if (!this.running) return;
    requestAnimationFrame(this._loop);
    const v = this.video;
    const vw = v.videoWidth, vh = v.videoHeight;
    if (!vw || !vh || v.readyState < 2) return;

    // カメラのコマは毎秒30枚。画面の更新（毎秒60回）のたびに処理すると、同じ絵を2回調べることになる
    const tick = performance.now();
    const watching = tick - (this.frameSeenAt || -1e9) < 500;
    if (watching ? !this.newFrame : tick - (this.lastProc || -1e9) < 30) return;
    this.newFrame = false;
    this.lastProc = tick;

    // 録画中でなければ、カメラの向きに合わせて画面サイズを決める
    if (!this.locked && (this.out.width !== vw || this.out.height !== vh)) {
      for (const c of [this.out, this.raw, this.overlay]) { c.width = vw; c.height = vh; }
      this._flushDelay();
      this.tracks = [];
      this.yQueue = [];
    }
    const W = this.out.width, H = this.out.height;

    // 1) カメラ映像を内部画面へ（録画中に向きが変わったら、枠内に収めて描く）
    if (vw === W && vh === H) {
      this.rctx.drawImage(v, 0, 0, W, H);
    } else {
      const s = Math.min(W / vw, H / vh);
      const dw = vw * s, dh = vh * s;
      this.rctx.fillStyle = '#000';
      this.rctx.fillRect(0, 0, W, H);
      this.rctx.drawImage(v, (W - dw) / 2, (H - dh) / 2, dw, dh);
    }

    const now = performance.now();
    this.nowTs = now;

    // 2) 顔検出
    let detected = false;
    if (this.detector) {
      try {
        this._grayUpdate(W, H);
        this._updateTracks(this._detect(W, H, now), now);
        this._applyYunet(now, W, H);
        this._applyPose(now, W, H);
        this._follow(now, W, H);
        this._mergeTracks(now);
        this._verify(now);
        detected = true;
        this._identify(now);
        this._scheduleYunet(now, W, H);
        this._schedulePose(now, W, H);
      } catch (e) {
        detected = false;
      }
    }

    // 3) いまのコマをためて、少し前のコマを録画される画面へ描く
    // カメラを切り替えた直後は、顔を探し終えるまで画面全体をぼかす
    if (this.coverFrom === -1) this.coverFrom = now;
    const cover = this.coverFrom > 0 && now - this.coverFrom < FLIP_COVER_MS;
    this._pushFrame(W, H, now, detected && !cover);
    this._drawDelayed(W, H, now);

    // 4) 目印（録画されない）
    this._drawOverlay(W, H);
  }

  // ---- 顔かどうかの調べ直し ----
  // 疑わしい枠は、その部分だけを拡大してもう一度検出にかける。本物の顔は拡大すると高い点数で
  // 検出され直すが、ライトや家具はほとんど検出されない。1コマに1つだけ調べる。
  _moving(now) { return now - (this.movedAt || -1e9) < MOVE_KEEP_MS; }

  _verify(now) {
    if (this.yunet && this.frameNo % VERIFY_EVERY_FRAMES) return;
    let pick = null;
    for (const t of this.tracks) {
      if (t.clear || t.seen !== now || now < t.nextV) continue;
      // まだ判断がついていない枠を先に、次に「顔ではない」とした枠、最後に「顔」と分かっている枠
      const rank = (x) => (x.fake ? 1 : x.verified ? 2 : 0);
      if (!pick || rank(t) < rank(pick) || (rank(t) === rank(pick) && t.nextV < pick.nextV)) pick = t;
    }
    if (!pick) return;
    let usedRot = pick.rot || 0;
    let score = this._cropScore(pick, usedRot, now);
    if (score < VERIFY_OK && !pick.strong && this.searchTurned && !this.yunet) {
      // まっすぐで顔と確かめられないときは、別の向き（横倒し・逆さ）でも1つずつ試す
      const alts = [0, 90, 270, 180].filter((r) => r !== (pick.rot || 0));
      const alt = alts[pick.altIdx++ % alts.length];
      const s2 = this._cropScore(pick, alt, now);
      // 別の向きは、はっきり顔と分かる点数のときだけ採用する（ライトなどが通りにくいように）
      if (s2 >= VERIFY_STRONG && s2 > score) { score = s2; usedRot = alt; pick.rot = alt; pick.view = null; pick.rotAt = now; pick.s = s2; }
    }
    // 最初にぼかさなかった枠で、顔らしい点数が出たら、ぼかしを入れて通常の調べ直しに移る
    if (pick.probation > 0) {
      pick.probation--;
      if (score >= BIRTH_OK) { pick.fake = false; pick.probation = 0; pick.vN = 0; pick.unsureN = 0; }
    }
    // 顔らしさがまったく無い状態が続いたら、顔はもうそこに無いと判断する。
    // （カメラをそらした直後、追跡が背景に取り残されて、ぼかしが残るのを防ぐ）
    // ただし、マスクの横顔などはこの調べ方では点数が出ないので、YuNetの手がかりも無いときだけにする
    pick.lowN = score < GONE_SCORE ? (pick.lowN || 0) + 1 : 0;
    const yHint = now - (pick.yAt || -1e9) < Y_GONE_MS;
    if (pick.lowN >= GONE_TIMES && !pick.fake && !yHint) {
      pick.strong = false; pick.vAvg = score; pick.vN = VERIFY_MIN_SAMPLES;
      pick.followAt = 0; pick.tpl = null;   // 見た目の追いかけもやめる
    }
    // 点数の平均を更新
    pick.vAvg = pick.vN ? pick.vAvg * 0.6 + score * 0.4 : score;
    pick.vN++;
    // 「ずっと顔として扱う」保護（横を向いたり、マスクで点数が下がったりしても、ぼかしを消さない）：
    //  ・まっすぐな向きで高い点数が出たら、1回で適用する（ライトなどは、まっすぐでは高い点数にならない）
    //  ・回転させた向きでは、高い点数が続いたときだけ適用する（ライトが回転でまぐれの高得点を出すことがあるため）
    if (usedRot === 0 ? score >= VERIFY_STRONG : (pick.vN >= 4 && pick.vAvg >= VERIFY_STRONG)) pick.strong = true;
    const turned = (pick.rot || 0) !== 0;
    const okLine = turned ? VERIFY_OK_TURNED : VERIFY_OK, ngLine = turned ? VERIFY_NG_TURNED : VERIFY_NG;
    const yOk = now - (pick.yOkAt || -1e9) < Y_RECENT_MS || now - (pick.poseAt || -1e9) < POSE_KEEP_MS;   // 別の検出が最近「顔」と判断した
    if (pick.strong || yOk || pick.vAvg >= okLine) {
      pick.verified = true; pick.fake = false; pick.unsureN = 0;
    } else {
      pick.verified = false;
      pick.unsureN = (pick.unsureN || 0) + 1;
      if ((pick.vN >= VERIFY_MIN_SAMPLES && pick.vAvg < ngLine) || pick.unsureN >= VERIFY_UNSURE_MAX) {
        if (!pick.fake) { pick.fake = true; this._rememberFake(pick, now); }
      }
      // どちらでもない間は、ぼかしたまま（「顔ではない」としていた枠は、そのまま）
    }
    // 「顔ではない」としていた枠で高い点数が出たら、人が入ってきたかもしれないので、すぐ次も調べる
    const hurry = pick.fake && score >= VERIFY_OK;
    pick.nextV = now + (hurry ? 0 : pick.probation > 0 || (pick.lowN > 0 && !pick.fake) ? VERIFY_EVERY_MS : pick.verified || pick.fake ? VERIFY_KEEP_MS : VERIFY_EVERY_MS);
  }

  // 枠の部分を、指定した向きに回して拡大し、もう一度検出にかけた点数を返す
  _cropScore(t, rot, now) {
    const S = 256, g = this.vctx2;
    const side = Math.max(t.w, t.h) * 2.4;
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2;
    const k = S / side;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#777';
    g.fillRect(0, 0, S, S);   // 画面の外にはみ出す部分は灰色のまま
    g.translate(S / 2, S / 2);
    g.rotate(rot * Math.PI / 180);
    g.scale(k, k);
    g.translate(-cx, -cy);
    g.drawImage(this.raw, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    let score = 0;
    for (const d of this.detector.detectForVideo(this.vCanvas, this._ts(now)).detections) {
      const b = d.boundingBox;
      if (!b) continue;
      // 切り出しの中央にある検出だけを数える（となりの人の顔を拾わないため）
      const dx = b.originX + b.width / 2 - S / 2, dy = b.originY + b.height / 2 - S / 2;
      if (Math.hypot(dx, dy) > S * 0.2 || b.width < S * 0.2) continue;
      score = Math.max(score, d.categories?.[0]?.score ?? 0);
    }
    return score;
  }

  // 「顔ではない」と分かった場所を覚える
  _rememberFake(t, now) {
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2;
    this.fakeZones = this.fakeZones.filter((z) => now - z.at < FAKE_ZONE_MS && Math.hypot(z.cx - cx, z.cy - cy) > z.w * 0.5);
    this.fakeZones.push({ cx, cy, w: t.w, at: now });
  }

  _inFakeZone(b, now) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    return this.fakeZones.some((z) => now - z.at < FAKE_ZONE_MS && Math.hypot(z.cx - cx, z.cy - cy) < Math.max(z.w, b.w) * 0.8);
  }

  // ---- 顔の見分け ----
  // 1回に1つの顔だけ、映像の合間に調べる（映像の動きを止めないため）
  _identify(now) {
    if (this.idBusy || !faceIdReady()) return;
    const havePersons = this.persons.some((p) => p.samples.length > 0);
    let pick = null;
    for (const t of this.tracks) {
      if (t.fake || now - t.kpAt > 120 || now < t.nextId) continue;   // いま見えている顔だけ
      if (t.clear ? !t.person : (!havePersons || t.noAuto)) continue;
      // ぼかし中の顔（戻ってきた人かもしれない）を優先
      if (!pick || (pick.clear && !t.clear) || (pick.clear === t.clear && t.nextId < pick.nextId)) pick = t;
    }
    if (!pick) return;
    pick.nextId = now + (pick.clear ? ID_CHECK_CLEAR_MS : ID_CHECK_BLUR_MS);
    if (!cropAligned(this.raw, pick.kp, this.idCanvas)) return;   // 横顔・小さい顔は判断しない
    const wasClear = pick.clear, person = pick.person;
    this.idBusy = true;
    describe(this.idCanvas).then((desc) => {
      if (!this.tracks.includes(pick) || pick.clear !== wasClear || pick.person !== person) return;
      if (pick.clear) this._checkClear(pick, desc); else this._checkBlurred(pick, desc);
    }).catch(() => {}).finally(() => { this.idBusy = false; });
  }

  // ぼかしなしの顔：角度ちがいを追加で覚える。別人に入れ替わっていたら、ぼかしに戻す
  _checkClear(t, desc) {
    const p = t.person;
    if (!this.persons.includes(p)) return;
    if (p.samples.length === 0) { p.samples.push(desc); return; }
    const d = nearest(p, desc);
    if (d >= ID_MISMATCH) {
      if (++t.misses >= ID_MISMATCH_TIMES) { t.clear = false; t.person = null; t.misses = 0; t.hits = 0; }
      return;
    }
    t.misses = 0;
    if (d > ID_LEARN_MIN && d < ID_MATCH && p.samples.length < ID_MAX_SAMPLES) p.samples.push(desc);
  }

  // ぼかし中の顔：選んだ人と確信できるときだけ、ぼかしを外す
  _checkBlurred(t, desc) {
    let best = null, bestD = Infinity;
    for (const p of this.persons) {
      if (!p.samples.length) continue;
      if (this.tracks.some((o) => o !== t && o.clear && o.person === p)) continue;   // その人はすでに映っている
      const d = nearest(p, desc);
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best && bestD < ID_MATCH) {
      if (t.candidate !== best) { t.candidate = best; t.hits = 0; }
      if (++t.hits >= ID_MATCH_TIMES) { t.clear = true; t.person = best; t.hits = 0; t.misses = 0; t.candidate = null; }
    } else {
      t.hits = 0; t.candidate = null;
    }
  }

  _ts(now) {
    const ts = Math.max(now, this.lastTs + 1);
    this.lastTs = ts;
    return ts;
  }

  _detect(W, H, now) {
    const boxes = [];
    // もう1つの顔検出が動いているときは、もとの顔検出の回数を減らす。
    // カメラを振っている間と、確かめている途中の枠があるときは、毎コマ調べる
    const yReady = !!this.yunet;
    const fno = (this.frameNo = (this.frameNo || 0) + 1);
    const urgent = this._moving(now) || this.tracks.some((t) => !t.fake && !t.clear && !t.verified);
    if (yReady && !urgent && fno % BF_EVERY_CALM) return boxes;
    const full = this.detector.detectForVideo(this.raw, this._ts(now));
    for (const d of full.detections) {
      const b = d.boundingBox;
      if (b) boxes.push({ x: b.originX, y: b.originY, w: b.width, h: b.height, rot: 0, full: true, s: d.categories?.[0]?.score ?? 0, kp: (d.keypoints || []).map((k) => ({ x: k.x * W, y: k.y * H })) });
    }
    // 追加の検出を1コマに1つずつ順番に行う：
    //  ・画面の一部を拡大（遠くの小さい顔を拾う）
    //  ・映像を回転（寝ている人など、横倒し・逆さの顔を拾う。顔検出はまっすぐな顔しか得意でないため）
    //  ・その両方（遠くで寝ている人の小さな顔を拾う）
    const tw = Math.round(W * 0.6), th = Math.round(H * 0.6);
    const pos = [[0, 0], [W - tw, 0], [0, H - th], [W - tw, H - th], [(W - tw) / 2, (H - th) / 2]];
    if (!this.views || this.viewsFor !== W + 'x' + H) {
      const tiles = pos.map(([tx, ty]) => ({ rot: 0, tx, ty, tw, th, q: 1 }));
      const turned = [];
      for (const rot of [90, 270]) {
        turned.push({ rot, tx: 0, ty: 0, tw: W, th: H, q: 0.5 });
        for (const [tx, ty] of pos) turned.push({ rot, tx, ty, tw, th, q: 1 });
      }
      turned.push({ rot: 180, tx: 0, ty: 0, tw: W, th: H, q: 0.5 });
      this.views = { tiles, turned };
      this.viewsFor = W + 'x' + H;
      this.slotIndex = 0;
    }
    const { tiles, turned } = this.views;
    // まっすぐの拡大と、回転とを1コマおきに。横倒しの顔を追っている間は、その顔を見つけた見方を優先する
    this.slotIndex++;
    const lying = this.tracks.find((t) => t.view && !t.fake && now - t.seen < 1500);
    let view = null;
    if (yReady) {
      // 回転して探す役目は、もう1つの顔検出が受け持つ。ここでは遠くの小さな顔だけを、ときどき拡大して探す
      if (fno % BF_EXTRA_EVERY === 0) view = tiles[this.slotIndex % tiles.length];
    } else if (this.slotIndex % 2 === 0 || !this.searchTurned) view = tiles[(this.searchTurned ? this.slotIndex >> 1 : this.slotIndex) % tiles.length];
    else view = turned[(this.slotIndex >> 1) % turned.length];
    if (view) this._detectView(view, boxes);
    // 追跡：画面全体の検出で毎コマ追えていない顔（寝ている顔・遠くの小さい顔）は、
    // その顔の周りだけを切り出して毎コマ調べ、位置を更新する（カメラが動いてもぼかしが遅れないように）
    let roi = 0;
    for (const t of this.tracks) {
      if (roi >= (yReady ? 1 : 3)) break;
      if (t.fake || now - t.seen > 1000 || now - (t.fullAt || 0) < 150) continue;
      if (yReady && now - (t.followAt || -1e9) < 100) continue;   // 見た目の追いかけで位置が分かっている顔は調べない
      roi++;
      const b = this._roiDetect(t, now);
      if (b) boxes.push(b);
    }
    return boxes;
  }

  // 画面の一部を拡大したり、回転したりして検出する
  _detectView(view, boxes) {
    const now = this.nowTs;
    const { rot, tx, ty, q } = view;
    const vw = view.tw, vh = view.th;
    const cw = Math.round((rot === 90 || rot === 270 ? vh : vw) * q), ch = Math.round((rot === 90 || rot === 270 ? vw : vh) * q);
    if (this.tile.width !== cw || this.tile.height !== ch) { this.tile.width = cw; this.tile.height = ch; }
    const g = this.tctx;
    if (rot === 90) g.setTransform(0, q, -q, 0, vh * q, 0);          // 時計回りに90度
    else if (rot === 270) g.setTransform(0, -q, q, 0, 0, vw * q);    // 反時計回りに90度
    else if (rot === 180) g.setTransform(-q, 0, 0, -q, vw * q, vh * q);
    else g.setTransform(q, 0, 0, q, 0, 0);
    g.drawImage(this.raw, -tx, -ty);
    g.setTransform(1, 0, 0, 1, 0, 0);
    // 回転・拡大した画面での座標 (u,v) を、元の画面の座標に戻す
    const back = (u, v) => {
      let X, Y;
      if (rot === 90) { X = v / q; Y = vh - u / q; }
      else if (rot === 270) { X = vw - v / q; Y = u / q; }
      else if (rot === 180) { X = vw - u / q; Y = vh - v / q; }
      else { X = u / q; Y = v / q; }
      return { x: X + tx, y: Y + ty };
    };
    for (const d of this.detector.detectForVideo(this.tile, this._ts(now)).detections) {
      const b = d.boundingBox;
      if (!b) continue;
      const p1 = back(b.originX, b.originY), p2 = back(b.originX + b.width, b.originY + b.height);
      boxes.push({ x: Math.min(p1.x, p2.x), y: Math.min(p1.y, p2.y), w: Math.abs(p2.x - p1.x), h: Math.abs(p2.y - p1.y), rot, view: rot ? view : null, s: d.categories?.[0]?.score ?? 0, kp: (d.keypoints || []).map((k) => back(k.x * cw, k.y * ch)) });
    }
  }

  // 顔の周りを、顔の向きに合わせて切り出して検出し、元の画面の座標に直した枠を返す
  _roiDetect(t, now) {
    const S = 256, g = this.vctx2;
    const p = this._predict(t, now);
    const side = Math.max(t.w, t.h) * 2.6;
    const cx = p.cx, cy = p.cy, k = S / side, th = (t.rot || 0) * Math.PI / 180;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#777';
    g.fillRect(0, 0, S, S);
    g.translate(S / 2, S / 2);
    g.rotate(th);
    g.scale(k, k);
    g.translate(-cx, -cy);
    g.drawImage(this.raw, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    const cos = Math.cos(th), sin = Math.sin(th);
    const back = (u, v) => {
      const dx = (u - S / 2) / k, dy = (v - S / 2) / k;
      return { x: cx + dx * cos + dy * sin, y: cy - dx * sin + dy * cos };
    };
    let best = null, bestS = 0.4;
    for (const d of this.detector.detectForVideo(this.vCanvas, this._ts(now)).detections) {
      const b = d.boundingBox, s = d.categories?.[0]?.score ?? 0;
      if (!b || s < bestS) continue;
      if (Math.hypot(b.originX + b.width / 2 - S / 2, b.originY + b.height / 2 - S / 2) > S * 0.3) continue;
      if (b.width < S * 0.2 || b.width > S * 0.75) continue;
      best = d; bestS = s;
    }
    if (!best) return null;
    const b = best.boundingBox;
    const c = back(b.originX + b.width / 2, b.originY + b.height / 2);
    const turned = t.rot === 90 || t.rot === 270;
    const w = (turned ? b.height : b.width) / k, h = (turned ? b.width : b.height) / k;
    return { x: c.x - w / 2, y: c.y - h / 2, w, h, roi: true, rot: t.rot || 0, view: t.view || null, s: bestS, kp: (best.keypoints || []).map((q) => back(q.x * S, q.y * S)) };
  }

  // 動きの速さから、いまの顔の中心を予測する
  _predict(t, now) {
    const dt = Math.min(Math.max(0, now - (t.upAt || now)), 250);
    return { cx: t.x + t.w / 2 + (t.vx || 0) * dt, cy: t.y + t.h / 2 + (t.vy || 0) * dt };
  }

  _updateTracks(boxes, now) {
    for (const b of boxes) {
      let best = null, bestScore = 0;
      for (const t of this.tracks) {
        const s = overlapScore(t, b);
        if (s > bestScore) { bestScore = s; best = t; }
      }
      if (best && bestScore > 0.15) {
        best.dAt = now;   // もう1つの検出（BlazeFace）がこの枠に反応した時刻
        // 同じ顔でも、向きのちがう見方（回転）がたまたま反応することがある。
        // 今の向きと同じ検出か、今の向きよりはっきり点数が高い検出か、今の向きでしばらく見えていないときだけ採用する
        const sameRot = (b.rot || 0) === (best.rot || 0);
        const adopt = sameRot || b.s > best.s + 0.15 || now - best.rotAt > 700;
        best.seen = now;
        if (adopt) {
          // 動きの速さ（1ミリ秒あたりの移動量）を測る
          const ncx = b.x + b.w / 2, ncy = b.y + b.h / 2, dt = now - best.upAt;
          // 検出のしかた（画面全体／追跡／拡大・回転）によって顔の中心が少しずれるので、
          // 同じしかたの検出どうしで比べる（ちがうしかたを比べると、止まっていても動いたように見える）
          const src = b.full ? 'full' : b.roi ? 'roi' : 'view';
          if (src === best.rsrc && dt > 0 && dt < 400) {
            best.vx = best.vx * 0.6 + ((ncx - best.rcx) / dt) * 0.4;
            best.vy = best.vy * 0.6 + ((ncy - best.rcy) / dt) * 0.4;
          } else if (dt >= 400) { best.vx = 0; best.vy = 0; }
          best.rcx = ncx; best.rcy = ncy; best.rsrc = src; best.upAt = now;
          if (b.full) best.fullAt = now;
          // 位置は素早く、大きさはゆっくり追う
          const a = 0.85, as = 0.5;
          best.w += (b.w - best.w) * as; best.h += (b.h - best.h) * as;
          best.x += (ncx - best.w / 2 - best.x) * a; best.y += (ncy - best.h / 2 - best.y) * a;
          best.kp = b.kp; best.kpAt = now;
          best.s = sameRot ? Math.max(b.s, best.s - 0.02) : b.s;   // 同じ向きの中では、最近の高い点数を覚えておく
          best.rot = b.rot || 0; best.view = b.view || null; best.rotAt = now;
          if (b.s >= 0.6 && !best.fake && !best.clear && this.gray) this._learnLook(best, this.gray, this.gw, this.gh, this.gk, best);
        }
        if (best.fake) {
          // はっきり顔らしい検出が来たら、すぐぼかしに戻す。
          // 位置や大きさが変わっただけなら、ぼかしは付けずに、すぐ調べ直す（カメラが動いただけのことが多い）
          const fb = best.fakeBox;
          const moved = !fb || Math.hypot(b.x + b.w / 2 - fb.cx, b.y + b.h / 2 - fb.cy) > fb.w * 0.3 || Math.abs(b.w - fb.w) > fb.w * 0.3;
          if (b.s >= STRONG_SCORE) { best.fake = false; best.vN = 0; best.vAvg = 0; best.nextV = 0; }
          else if (moved && b.s >= PROV_MIN && this._moving(now) && b.w <= this.out.width * PROV_MAX_W) { best.fake = false; best.vN = 0; best.vAvg = 0; best.nextV = 0; best.unsureN = 0; best.provAt = now; }
          else if (moved) { best.nextV = 0; best.fakeBox = { cx: b.x + b.w / 2, cy: b.y + b.h / 2, w: b.w }; this._rememberFake(best, now); }
        }
      } else if (!this.tracks.some((t) => !t.fake && t.seen === now && coverage(t, b) > 0.3)) {
        // いま見えている顔と大きく重なる検出は、同じ顔の二重検出なので新しい顔として扱わない
        const nt = { id: this.nextId++, dAt: now, x: b.x, y: b.y, w: b.w, h: b.h, seen: now, clear: false, vx: 0, vy: 0, upAt: now, rcx: b.x + b.w / 2, rcy: b.y + b.h / 2, fullAt: b.full ? now : 0, rot: b.rot || 0, view: b.view || null, altIdx: 0, s: b.s, rotAt: now, kp: b.kp, kpAt: now, person: null, nextId: 0, hits: 0, misses: 0, noAuto: false, verified: false, fake: b.s < STRONG_SCORE && !this._moving(now) && this._inFakeZone(b, now), vAvg: 0, vN: 0, strong: false, nextV: 0, fakeBox: { cx: b.x + b.w / 2, cy: b.y + b.h / 2, w: b.w } };
        // はっきり顔と分かる検出（点数が高い）は、最初のコマからぼかす。
        // 怪しい検出は、ぼかす前にその場で拡大して調べ、顔らしさがまったく無ければ最初からぼかさない
        // （人がいない場面で、物にぼかしが一瞬出るのを防ぐ）。ぼかさなかった枠も、しばらくは短い間隔で調べ続ける
        if (!nt.fake && b.s < STRONG_SCORE) {
          let s0 = this._cropScore(nt, nt.rot, now);
          const birthLine = this.yunet ? BIRTH_OK_Y : BIRTH_OK;
          if (s0 < birthLine && this.searchTurned && !this.yunet) {
            for (const r of [0, 90, 270, 180]) {
              if (r === nt.rot) continue;
              s0 = Math.max(s0, this._cropScore(nt, r, now));
              if (s0 >= BIRTH_OK) break;
            }
          }
          // カメラを動かしている間は、顔が画面に入ってきた直後のことが多いので、調べがつくまで仮にぼかしておく
          // （マスクの寝顔などは、ここでは顔らしい点数が出ない。もう1つの検出が確かめるまでの間、素顔が映らないように）
          const prov = s0 < birthLine && this._moving(now) && b.s >= PROV_MIN && b.w <= this.out.width * PROV_MAX_W;
          if (prov) nt.provAt = now;
          else if (s0 < birthLine) { nt.fake = true; nt.probation = BIRTH_RETRIES; nt.nextV = now + VERIFY_EVERY_MS; }
          if (this.onBirth) this.onBirth({ s: b.s, s0, rot: nt.rot, cx: Math.round(b.x + b.w / 2), cy: Math.round(b.y + b.h / 2), w: Math.round(b.w) });
        }
        if (!nt.fake && this.gray) this._learnLook(nt, this.gray, this.gw, this.gh, this.gk, nt);
        this.tracks.push(nt);
      }
    }
    // 仮にぼかした枠で、もう1つの検出が確かめられなかったものは外す
    for (const t of this.tracks) {
      if (t.provAt && !t.fake && !t.clear && now - t.provAt > (t.poseProv ? POSE_TWICE_MS : PROV_MS)) {
        if ((t.yOkAt || 0) >= t.provAt || t.strong || t.verified) { t.provAt = 0; continue; }
        t.provAt = 0; t.fake = true; t.verified = false; this._rememberFake(t, now);
      }
    }
    // 見失った顔の整理。ぼかさない顔は、見失ったら選択を解除する（次に現れたらぼかす）
    const W = this.out.width, H = this.out.height;
    this.tracks = this.tracks.filter((t) => {
      if (t.clear) return now - t.seen < HOLD_CLEAR_MS;
      const cx = t.x + t.w / 2, cy = t.y + t.h / 2;
      // 画面の外へ出た。顔の中心が端を越えても、顔の一部はまだ映っていることがあるので、枠がほぼ全部出るまで残す
      if (cx < -t.w * 0.45 || cy < -t.h * 0.45 || cx > W + t.w * 0.45 || cy > H + t.h * 0.45) return false;
      const m = t.w * 0.25;
      const atEdge = t.x < m || t.y < m || t.x + t.w > W - m || t.y + t.h > H - m;
      // 見た目で追いかけられている顔は、検出が途切れていても残す（マスクの横顔などは検出が途切れやすい）
      const firmFace = t.strong || (t.yBest || 0) >= Y_FIRM;
      const followed = (t.followAt && now - t.followAt < 200 && now - Math.max(t.seen, t.yAt || 0) < (firmFace ? FOLLOW_MAX_MS : FOLLOW_WEAK_MS))
        // 画面の端からはみ出しかけている顔。最後に検出されてから EDGE_HOLD_MS までに限る（顔が去ったあと端にぼかしが残らないように）
        || (t.verified && !t.fake && t.edgeAt && now - t.edgeAt < 200 && now - Math.max(t.seen, t.yAt || 0, t.poseAt || 0) < EDGE_HOLD_MS);
      if (followed || (!t.provAt && now - (t.poseAt || -1e9) < POSE_KEEP_MS)) return true;
      if (t.provAt && t.poseProv) return now - t.provAt < POSE_TWICE_MS;
      // 点数の低い枠は、カメラを動かしている間に見失ったらすぐ消す
      if (!firmFace && this._moving(now)) return now - t.seen < HOLD_SWING_MS;
      if (!t.verified) return now - t.seen < HOLD_UNSURE_MS;
      if (!firmFace) return now - t.seen < HOLD_WEAK_MS;
      return now - t.seen < (atEdge ? HOLD_EDGE_MS : HOLD_BLUR_MS);
    });  }

  // ---- 同じ顔に重なった枠を1つにまとめる ----
  // 2つの検出が別々に枠を作ると、同じ顔にぼかしの枠が二重・三重にかかり、タップもしにくくなる
  _mergeTracks(now) {
    const rank = (t) => (t.clear ? 5 : t.fake ? 0 : now - (t.yOkAt || -1e9) < Y_RECENT_MS ? 4 : t.strong ? 3 : t.verified ? 2 : 1);
    for (let again = true; again;) {
      again = false;
      for (let i = 0; i < this.tracks.length && !again; i++) {
        for (let j = i + 1; j < this.tracks.length && !again; j++) {
          const a = this.tracks[i], b = this.tracks[j];
          if (!sameFace(a, b)) continue;
          const ra = rank(a), rb = rank(b);
          // 同じ確かさなら、小さいほう（顔にぴったり合っている枠）を残す
          const keep = ra > rb || (ra === rb && a.w <= b.w) ? a : b, drop = keep === a ? b : a;
          absorbTrack(keep, drop);
          drop.mergedInto = keep;
          this.tracks.splice(this.tracks.indexOf(drop), 1);
          again = true;
        }
      }
    }
  }

  // ---- 見た目の追いかけ ----
  // 毎コマ、映像を小さな白黒の画面にする（追いかけ用。録画はされない）
  _grayUpdate(W, H) {
    const gw = GRAY_W, gh = Math.max(2, Math.round(GRAY_W * H / W));
    const c = this.grayCanvas;
    if (c.width !== gw || c.height !== gh) { c.width = gw; c.height = gh; }
    this.gctx.drawImage(this.raw, 0, 0, W, H, 0, 0, gw, gh);
    const d = this.gctx.getImageData(0, 0, gw, gh).data;
    if (!this.gray || this.gray.length !== gw * gh) {
      this.gray = new Float32Array(gw * gh);
      this.grayPrev = null;
    }
    // 1つ前のコマを取っておく（カメラの動きを測るため）
    const prev = this.grayPrev && this.grayPrev.length === this.gray.length ? this.grayPrev : new Float32Array(gw * gh);
    const hadPrev = !!this.grayPrev && this.grayPrev.length === this.gray.length && this.grayReady;
    this.grayPrev = this.gray;
    this.gray = prev;
    const g = this.gray, p = this.grayPrev;
    let diff = 0;
    for (let i = 0, j = 0; i < g.length; i++, j += 4) {
      const v = d[j] * 0.299 + d[j + 1] * 0.587 + d[j + 2] * 0.114;
      diff += Math.abs(v - p[i]);
      g[i] = v;
    }
    this.grayReady = true;
    // カメラを動かしているかどうか（画面全体の明るさの変化で判断する）
    this.moveDiff = diff / g.length;
    if (this.moveDiff >= MOVE_DIFF) this.movedAt = this.nowTs;
    this.gw = gw; this.gh = gh; this.gk = gw / W;
    if (hadPrev) this._shiftWithCamera(this._cameraShift(p, g, gw, gh), gw / W);
  }

  // 1つ前のコマからいまのコマまでに、映像全体がどれだけずれたかを測る（縮小画面の点の数で返す）。
  // まず粗い画面で大まかに探し、次に細かい画面で合わせる。模様が少なく判断できないときは null
  _cameraShift(prev, cur, gw, gh) {
    const S = 4, cw = gw >> 2, ch = gh >> 2, n = cw * ch;
    if (!this.gmA || this.gmA.length !== n) { this.gmA = new Float32Array(n); this.gmB = new Float32Array(n); }
    const A = this.gmA, B = this.gmB;
    for (let y = 0, k = 0; y < ch; y++) for (let x = 0; x < cw; x++, k++) {
      let a = 0, b = 0;
      for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
        const q = (y * S + j) * gw + x * S + i;
        a += prev[q]; b += cur[q];
      }
      A[k] = a / 16; B[k] = b / 16;
    }
    const R = GM_RANGE;
    let best = Infinity, bx = 0, by = 0, sum = 0, cnt = 0;
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      let e = 0, m = 0;
      for (let y = R; y < ch - R; y += 2) for (let x = R; x < cw - R; x += 2) {
        e += Math.abs(A[y * cw + x] - B[(y + dy) * cw + x + dx]);
        m++;
      }
      e /= m; sum += e; cnt++;
      if (e < best) { best = e; bx = dx; by = dy; }
    }
    const mean = sum / cnt;
    if (mean < GM_MIN_TEXTURE || best > mean * GM_MAX_RATIO) return null;   // 白い壁など、ずれを測れない場面
    // 細かい画面で合わせる
    let fine = Infinity, fx = bx * S, fy = by * S;
    const M = R * S + S;
    for (let dy = by * S - 3; dy <= by * S + 3; dy++) for (let dx = bx * S - 3; dx <= bx * S + 3; dx++) {
      let e = 0, m = 0;
      for (let y = M; y < gh - M; y += 5) for (let x = M; x < gw - M; x += 5) {
        e += Math.abs(prev[y * gw + x] - cur[(y + dy) * gw + x + dx]);
        m++;
      }
      e /= m;
      if (e < fine) { fine = e; fx = dx; fy = dy; }
    }
    return { dx: fx, dy: fy };
  }

  // カメラが動いた分だけ、すべての枠を映像と一緒に動かす。これで、見失った顔にもぼかしが付いていく
  _shiftWithCamera(m, gk) {
    if (!m || (!m.dx && !m.dy)) return;
    const dx = m.dx / gk, dy = m.dy / gk;
    this.camX = (this.camX || 0) + dx;
    this.camY = (this.camY || 0) + dy;
    for (const t of this.tracks) {
      t.x += dx; t.y += dy;
      if (t.rcx !== undefined) { t.rcx += dx; t.rcy += dy; }
      if (t.fakeBox) { t.fakeBox.cx += dx; t.fakeBox.cy += dy; }
    }
    for (const z of this.fakeZones) { z.cx += dx; z.cy += dy; }
    for (const c of this.ySoft || []) { c.cx += dx; c.cy += dy; }
    for (const c of this.yProps || []) { c.cx += dx; c.cy += dy; }
    for (const c of this.yRejects || []) { c.cx += dx; c.cy += dy; }
    for (const c of this.poseCands || []) { c.cx += dx; c.cy += dy; }
    if (this.personBox) { this.personBox.cx += dx; this.personBox.cy += dy; }
  }

  // 顔の見た目（白黒の点の並び）を覚える。box は元の画面の座標
  _makeTemplate(gray, gw, gh, gk, box) {
    const bw = box.w * gk, bh = box.h * gk;
    if (bw < 4 || bh < 4) return null;
    const nx = Math.min(16, Math.max(4, Math.round(bw))), ny = Math.min(16, Math.max(4, Math.round(bh)));
    const n = nx * ny, offs = new Int32Array(n), v = new Float32Array(n);
    const x0 = Math.round(box.x * gk), y0 = Math.round(box.y * gk);
    let k = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      offs[k++] = Math.round((j + 0.5) * bh / ny) * gw + Math.round((i + 0.5) * bw / nx);
    }
    const iw = Math.ceil(bw) + 1, ih = Math.ceil(bh) + 1;
    if (x0 < 0 || y0 < 0 || x0 + iw >= gw || y0 + ih >= gh) return null;   // 画面の端にかかる顔は覚えない
    let sum = 0;
    for (k = 0; k < n; k++) { v[k] = gray[y0 * gw + x0 + offs[k]]; sum += v[k]; }
    const mean = sum / n;
    let ss = 0;
    for (k = 0; k < n; k++) { v[k] -= mean; ss += v[k] * v[k]; }
    if (ss / n < 9) return null;   // 模様がほとんど無い（白い壁など）は追えない
    return { v, offs, n, iw, ih, norm: Math.sqrt(ss), bw: box.w, bh: box.h };
  }

  // 覚えた見た目を、(cx,cy) の周り（元の画面の座標、半径 R）で探す。見つかった中心と似ている度合いを返す
  _matchTemplate(tpl, gray, gw, gh, gk, cx, cy, R) {
    const r = Math.max(2, Math.round(R * gk));
    const pcx = Math.round(cx * gk - tpl.iw / 2), pcy = Math.round(cy * gk - tpl.ih / 2);
    const xmin = Math.max(0, pcx - r), xmax = Math.min(gw - tpl.iw - 1, pcx + r);
    const ymin = Math.max(0, pcy - r), ymax = Math.min(gh - tpl.ih - 1, pcy + r);
    if (xmin > xmax || ymin > ymax) return null;
    const { v, offs, n, norm } = tpl;
    const score = (px, py) => {
      const base = py * gw + px;
      let s = 0, s2 = 0, st = 0;
      for (let k = 0; k < n; k++) { const a = gray[base + offs[k]]; s += a; s2 += a * a; st += a * v[k]; }
      const vv = s2 - s * s / n;
      return vv < n * 4 ? -1 : st / (norm * Math.sqrt(vv));
    };
    let best = -2, bx = pcx, by = pcy;
    for (let py = ymin; py <= ymax; py += 2) for (let px = xmin; px <= xmax; px += 2) {
      const s = score(px, py);
      if (s > best) { best = s; bx = px; by = py; }
    }
    // 細かく合わせる
    const cx0 = bx, cy0 = by;
    for (let py = cy0 - 1; py <= cy0 + 1; py++) for (let px = cx0 - 1; px <= cx0 + 1; px++) {
      if (px < xmin || px > xmax || py < ymin || py > ymax || (px === cx0 && py === cy0)) continue;
      const s = score(px, py);
      if (s > best) { best = s; bx = px; by = py; }
    }
    // 画面の端で探す範囲が削られていて、一番似ている場所がその削られた側の端にあるときは、
    // 顔が画面の外へはみ出しかけている（本当の位置は画面の外側）。内側の似た模様を顔とまちがえやすい
    const atEdge = (xmax < pcx + r && bx >= xmax - 1) || (xmin > pcx - r && bx <= xmin + 1)
      || (ymax < pcy + r && by >= ymax - 1) || (ymin > pcy - r && by <= ymin + 1);
    return { cx: (bx + tpl.iw / 2) / gk, cy: (by + tpl.ih / 2) / gk, ncc: best, atEdge };
  }

  // 検出が来なかったコマでも、覚えた見た目を探して、ぼかしの位置を動かす
  _follow(now) {
    if (!this.gray) return;
    for (const t of this.tracks) {
      if (t.clear || t.fake || !t.verified || t.upAt === now) continue;
      const p = this._predict(t, now);
      const R = Math.max(t.w * 0.6, 24);
      const W = this.out.width, H = this.out.height, e = t.w * 0.15, j = t.w * 0.25;
      const nearEdge = p.cx + t.w / 2 > W - e || p.cx - t.w / 2 < e || p.cy + t.h / 2 > H - e || p.cy - t.h / 2 < e;
      // 見た目を覚えていない顔（画面の端で見つかった顔など）は、見た目では追えない
      const m = t.tpl ? this._matchTemplate(t.tpl, this.gray, this.gw, this.gh, this.gk, p.cx, p.cy, R) : null;
      // 画面の端にかかっている顔が、内側の別の場所へ大きく飛ぼうとしたら、それも「はみ出しかけ」とみなす
      // （半分切れた顔は見た目が合わず、近くの似た模様〔服・ぬいぐるみなど〕へ移りやすい）。
      // 端ぎりぎりで探す範囲が画面の外になり、見た目を照らし合わせられないときも同じ
      const jumpIn = m && nearEdge && Math.hypot(m.cx - p.cx, m.cy - p.cy) > j;
      if ((m && (m.atEdge || jumpIn)) || (!m && nearEdge)) {
        // 顔が画面の端からはみ出しかけている。内側へは動かさず、いまの位置（はみ出した位置）でぼかし続ける。
        // 本当に画面の外へ出たときは、カメラの動きと一緒に枠も画面の外へ出ていく
        t.edgeAt = now;   // 残す長さは、見失った顔の整理（_updateTracks）で決める
        t.vx = (t.vx || 0) * 0.8; t.vy = (t.vy || 0) * 0.8;   // 見込みで外へ流れていかないように
        // カメラの動きと一緒に外へ流れて「画面の外へ出た」と消されないよう、枠の4分の1は画面の中に残す
        // （本当に去った顔は、最後の検出から EDGE_HOLD_MS で終わる）
        t.x = Math.max(-t.w * 0.75, Math.min(W - t.w * 0.25, t.x));
        t.y = Math.max(-t.h * 0.75, Math.min(H - t.h * 0.25, t.y));
        continue;
      }
      if (!m || m.ncc < NCC_OK) continue;   // 見た目を覚えていない顔も、ここで終わり（これまでどおり）
      const dt = now - t.upAt;
      if (t.rsrc === 'ncc' && dt > 0 && dt < 400) {
        t.vx = t.vx * 0.6 + ((m.cx - t.rcx) / dt) * 0.4;
        t.vy = t.vy * 0.6 + ((m.cy - t.rcy) / dt) * 0.4;
      }
      t.rcx = m.cx; t.rcy = m.cy; t.rsrc = 'ncc'; t.upAt = now;
      t.x = m.cx - t.w / 2; t.y = m.cy - t.h / 2;
      t.followAt = now;
    }
  }

  // 検出で位置が確かな顔は、そのときの見た目を覚え直す
  _learnLook(t, gray, gw, gh, gk, box) {
    const tpl = this._makeTemplate(gray, gw, gh, gk, box);
    if (tpl) t.tpl = tpl;
  }

  // ---- もう1つの顔検出（YuNet） ----
  // 1回に1つの向きで、別の流れで調べる。結果が返ってくるまでの間に動いた分は、見た目の追いかけで補正する
  // 追っている顔があれば、「画面全体を調べる回」と「顔の周りを4区画で一度に調べる回」を交互に行う。
  // 顔の周りは拡大して調べるので、横顔・マスクの顔でも点数が出やすく、向きを変えて何通りも試せる
  // 2つ目が読み込めていれば、2つを同時に動かす（1つは画面全体の見張り専用）。新しく入ってきた顔を早く見つけるため
  // カメラを振っている間（顔が画面に入ってくる瞬間）だけ全力で探し、落ち着いているときは休みながら調べる
  _scheduleYunet(now, W, H) {
    if (!this.gray || !this.yunet) return;
    if (!this.yStartAt) this.yStartAt = now;
    const swinging = now - (this.movedAt || -1e9) < Y_HOT_MS;
    const starting = now - this.yStartAt < Y_START_HOT_MS;
    const waiting = now < (this.yAgainUntil || 0) || (this.yProps || []).length > 0 || this.tracks.some((t) => !t.fake && !t.clear && !t.verified);   // 顔かどうか確かめている途中の枠がある
    const newView = Math.hypot((this.camX || 0) - (this.scanX || 0), (this.camY || 0) - (this.scanY || 0)) > W * Y_NEW_VIEW;   // 前回の検査のあと、新しい景色が入った
    const noFace = !this.tracks.some((t) => !t.fake && t.verified);
    if (swinging) { if (!this.twinFrom) this.twinFrom = now; } else this.twinFrom = 0;
    const twin = !!this.yunet2 && swinging && now - this.twinFrom < Y_TWIN_MAX_MS;
    const rest = Math.max(noFace ? Y_IDLE_GAP_MS : Y_CALM_GAP_MS, (this.yMs || 0) * Y_CALM_DUTY);
    if (!this.yBusy && (swinging || starting || waiting || newView || now - (this.yEndAt || -1e9) >= rest)) {
      this._runYunet(this.yunet, 'yBusy', now, W, H, twin ? 'full' : 'any');
    }
    if (twin && !this.yBusy2) this._runYunet(this.yunet2, 'yBusy2', now, W, H, 'any');
  }

  _runYunet(det, busyKey, now, W, H, mode) {
    // 顔の周りを調べる対象：ぼかし中の顔を先に。「顔ではない」とした枠も、いま検出が出ているものは
    // 調べ直す（マスクの寝顔が「顔ではない」と判定されたままになるのを防ぐ）
    const rank = (t) => (t.verified && !t.fake ? 0 : !t.fake ? 1 : 2);
    const targets = this.tracks
      .filter((t) => !t.clear && now - Math.max(t.seen, t.followAt || 0) < (t.fake ? 300 : 1500))
      .sort((a, b) => rank(a) - rank(b))
      .slice(0, 4);
    let cells, kind;
    // 顔の周りを調べるのは、新しい部品でまだ確かめていない枠があるときと、確かめた顔でも少し時間がたったときだけ。
    // それ以外は画面全体を調べる回を増やし、新しく画面に入ってきた顔を早く見つける
    const needRoi = targets.some((t) => now - (t.yOkAt || -1e9) > (t.verified && !t.fake ? 400 : 0));
    const props = mode === 'full' ? [] : this._takeProps(now);
    const unsure = targets.some((t) => !t.verified && !t.fake);
    if (props.length) {
      // 候補の場所を拡大して調べ直す。候補の向きと、その前後の向きで試す
      kind = 'roi';
      const plan = [];
      const per = props.length === 1 ? [0, -45, 45, 90] : [0, 45];
      for (const p of props) for (const d of per) plan.push({ p, rot: (p.rot + d + 360) % 360 });
      cells = plan.slice(0, 4).map(({ p, rot }, i) => ({ cx: p.cx, cy: p.cy, side: Math.max(p.w, p.h) * 2.4, rot, ox: (i % 2) * 320, oy: (i >> 1) * 320, size: 320, prop: p }));
    } else if (mode !== 'full' && targets.length && needRoi && (unsure || this.yIdx++ % 2 === 1)) {
      kind = 'roi';
      const rots = this.searchTurned ? [90, 270, 0, 315, 45] : [0, 330, 30];
      // 区画を顔の数で分け、それぞれの顔を「いちばん反応した向き」と「まだ試していない向き」で調べる
      const plan = [];
      const per = Math.floor(4 / targets.length), extra = 4 % targets.length;
      targets.forEach((t, i) => {
        const n = per + (i < extra ? 1 : 0);
        const mine = [];
        if (rots.includes(t.yRot) && now - (t.yRotAt || 0) < 2000) mine.push(t.yRot);
        for (let k = 0; mine.length < n && k < rots.length; k++) {
          const r = rots[(t.yCyc = ((t.yCyc || 0) + 1)) % rots.length];
          if (!mine.includes(r)) mine.push(r);
        }
        for (const r of mine) plan.push({ t, rot: r });
      });
      cells = plan.map(({ t, rot }, i) => {
        const p = this._predict(t, now);
        return { cx: p.cx, cy: p.cy, side: Math.max(t.w, t.h) * 2.4, rot, ox: (i % 2) * 320, oy: (i >> 1) * 320, size: 320, track: t };
      });
    } else {
      kind = 'full';
      const n = (this.yFull = (this.yFull || 0) + 1);
      this.scanX = this.camX || 0; this.scanY = this.camY || 0;
      if (this.searchTurned && n % 4 !== 0) {
        // 画面全体を4つの向き（まっすぐ・右倒し・左倒し・斜め）で同時に調べる。寝ている顔を1回で見つけるため
        const diag = [315, 45, 225, 135][n % 4];
        cells = [0, 90, 270, diag].map((rot, i) => ({ ...YuNetDetector.fullCell(W, H, rot), ox: (i % 2) * 320, oy: (i >> 1) * 320, size: 320 }));
      } else {
        // 4回に1回は、まっすぐの向きだけを大きく調べる（遠くの小さな顔のため）
        cells = [YuNetDetector.fullCell(W, H, 0)];
      }
    }
    this[busyKey] = true;
    const job = { at: now, kind, cells, gray: this.gray.slice(), gw: this.gw, gh: this.gh, gk: this.gk, W, H, camX: this.camX || 0, camY: this.camY || 0 };
    const began = performance.now();
    const gen = this.camGen;
    det.detect(this.raw, cells.map(({ track, prop, ...c }) => c))
      .then((boxes) => { if (this.running && gen === this.camGen) this.yQueue.push({ job, boxes }); })
      .catch(() => {})
      .finally(() => {
        this[busyKey] = false;
        this.yEndAt = performance.now();
        this.yMs = this.yMs ? this.yMs * 0.8 + (this.yEndAt - began) * 0.2 : this.yEndAt - began;
      });
  }

  // 画面全体の検査で見つけた候補を覚える（あとで拡大して調べ直す）
  _propose(b, now) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    this.yProps = this.yProps || [];
    this.yRejects = (this.yRejects || []).filter((r) => now - r.at < Y_PROP_NG_MS);
    const near = (c) => Math.hypot(c.cx - cx, c.cy - cy) < Math.max(c.w, b.w) * 0.6;
    if (this.yProps.some(near) || this.yRejects.some(near)) return;
    this.yProps.push({ cx, cy, w: b.w, h: b.h, rot: b.rot, s: b.s, at: now, hit: false });
  }

  // 調べ直す候補を、点数の高い順に2つまで取り出す
  _takeProps(now) {
    this.yProps = (this.yProps || []).filter((p) => now - p.at < Y_PROP_LIFE_MS);
    this.yProps.sort((a, b) => b.s - a.s);
    return this.yProps.splice(0, 2);
  }

  // もう1つの顔検出で見つけた顔の、新しい枠を作る
  _birthY(b, firm, now) {
    const nt = { id: this.nextId++, x: b.x, y: b.y, w: b.w, h: b.h, seen: now, clear: false, vx: 0, vy: 0, upAt: now, rcx: b.x + b.w / 2, rcy: b.y + b.h / 2, rsrc: 'y', fullAt: 0, rot: b.rot, view: null, altIdx: 0, s: b.s, rotAt: now, kp: yunetKp(b.kp), kpAt: now, person: null, nextId: 0, hits: 0, misses: 0, noAuto: false, verified: firm, provAt: firm ? 0 : now, fake: false, vAvg: 0, vN: 0, strong: false, nextV: now + VERIFY_EVERY_MS, yBest: b.s, yAt: now, yOkAt: firm ? now : 0, yRot: b.rot, yRotS: b.s, yRotAt: now, fakeBox: null };
    this._learnLook(nt, this.gray, this.gw, this.gh, this.gk, nt);
    this.tracks.push(nt);
    if (this.onBirth) this.onBirth({ s: b.s, s0: b.s, rot: b.rot, cx: Math.round(b.x + b.w / 2), cy: Math.round(b.y + b.h / 2), w: Math.round(b.w), y: true });
    return nt;
  }

  // ---- 姿勢から頭の位置を割り出す ----
  _schedulePose(now, W, H) {
    if (!this.pose || this.poseBusy) return;
    const nearFace = this.tracks.some((t) => !t.fake && t.verified && t.w > W * POSE_BIG_FACE);
    const swinging = now - (this.movedAt || -1e9) < Y_HOT_MS;
    const gap = Math.max(swinging ? POSE_GAP_SWING_MS : nearFace ? POSE_GAP_BUSY_MS : POSE_GAP_MS, (this.poseMs || 0) * POSE_DUTY);
    if (now - (this.poseEnd || -1e9) < gap) return;
    this.poseBusy = true;
    // 画面全体と、見つけた人の周りの拡大を、交互に調べる
    this.poseTurn = !this.poseTurn;
    const crop = this.poseTurn ? this._personCrop(now, W, H) : null;
    const job = { camX: this.camX || 0, camY: this.camY || 0, W, H, crop };
    const began = performance.now();
    const gen = this.camGen;
    this.pose.detect(this.raw, crop)
      .then((people) => { if (this.running && gen === this.camGen) this.pQueue.push({ job, people }); })
      .catch(() => {})
      .finally(() => {
        this.poseBusy = false;
        this.poseEnd = performance.now();
        this.poseMs = this.poseMs ? this.poseMs * 0.8 + (this.poseEnd - began) * 0.2 : this.poseEnd - began;
      });
  }

  // 見つけた人の周りの、拡大して調べる範囲
  _personCrop(now, W, H) {
    const p = this.personBox;
    if (!p || now - p.at > POSE_PERSON_MS) return null;
    const side = Math.max(p.w, p.h, 200) * POSE_ZOOM;
    const w = Math.round(Math.min(side, W)), h = Math.round(Math.min(side, H));
    if (w >= W * 0.9 && h >= H * 0.9) return null;
    const x = Math.round(Math.max(0, Math.min(W - w, p.cx - w / 2))), y = Math.round(Math.max(0, Math.min(H - h, p.cy - h / 2)));
    return { x, y, w, h };
  }

  // 同じ場所を少し前にも指していれば true
  _poseTwice(cx, cy, r, now) {
    this.poseCands = this.poseCands.filter((c) => now - c.at < POSE_TWICE_MS);
    const hit = this.poseCands.some((c) => c.at < now && Math.hypot(c.cx - cx, c.cy - cy) < Math.max(c.r, r) * 1.5);
    this.poseCands.push({ cx, cy, r, at: now });
    return hit;
  }

  _applyPose(now, W, H) {
    if (!this.pQueue.length) return;
    for (const { job, people } of this.pQueue.splice(0)) {
      if (job.W !== W || job.H !== H) continue;
      // 調べている間にカメラが動いた分を足す
      const dx = (this.camX || 0) - job.camX, dy = (this.camY || 0) - job.camY;
      const c = job.crop;
      for (const raw of people) {
        // 拡大して調べたときは、画面全体の座標に戻す
        const lm = c ? raw.map((p) => ({ x: (c.x + p.x * c.w) / W, y: (c.y + p.y * c.h) / H, v: p.v })) : raw;
        const head = poseHead(lm, W, H);
        if (!head) continue;
        const cx = head.cx + dx, cy = head.cy + dy, r = head.r;
        // 人の範囲を覚える
        const seen = lm.filter((p) => p.v > 0.5);
        const xs = seen.map((p) => p.x * W), ys = seen.map((p) => p.y * H);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
        this.personBox = { cx: (x0 + x1) / 2 + dx, cy: (y0 + y1) / 2 + dy, w: x1 - x0, h: y1 - y0, at: now };
        let best = null, bestD = Infinity;
        for (const t of this.tracks) {
          const d = Math.hypot(t.x + t.w / 2 - cx, t.y + t.h / 2 - cy);
          if (d < Math.max(t.w * 0.9, r * 1.6) && d < bestD) { bestD = d; best = t; }
        }
        const twice = this._poseTwice(cx, cy, r, now);
        if (best) {
          best.poseAt = now;
          best.yAt = now;
          if (best.clear) continue;
          if ((best.fake || !best.verified) && twice) {
            best.fake = false; best.verified = true; best.probation = 0; best.unsureN = 0; best.provAt = 0;
            best.yOkAt = now; best.yBest = Math.max(best.yBest || 0, Y_FIRM);
          }
          // 検出で位置が更新されていない顔は、頭の位置へ寄せる
          // 1回だけの外れた反応で枠を動かさない（柄のある服の胸を頭と取りちがえることがあるため）
          const off = bestD > r * 0.7 && twice && now - (best.yOkAt || -1e9) > 600;
          if (off || (bestD <= r * 0.7 && now - (best.upAt || -1e9) > 400)) {
            const a = off ? 0.85 : 0.6;
            best.x += (cx - best.w / 2 - best.x) * a; best.y += (cy - best.h / 2 - best.y) * a;
            best.rcx = cx; best.rcy = cy; best.rsrc = 'pose'; best.upAt = now;
            if (off && this.gray) this._learnLook(best, this.gray, this.gw, this.gh, this.gk, best);
          }
        } else {
          // 1回目の反応で仮にぼかす。2回目が来なければ取り消す（映像を遅らせているので、録画前に消せる）
          const w = r * POSE_BOX * 2;
          const nt = this._birthY({ x: cx - w / 2, y: cy - w / 2, w, h: w, s: Y_FIRM, rot: 0, kp: [] }, twice, now);
          nt.poseAt = now;
          nt.poseProv = !twice;
          nt.kpAt = 0;
        }
      }
    }
  }

  // 同じ場所で少し前にも反応していれば true。1回きりの反応（ライトや模様のまぐれ）をふるい落とす
  _yTwice(b, now) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    this.ySoft = (this.ySoft || []).filter((c) => now - c.at < Y_TWICE_MS);
    const hit = this.ySoft.some((c) => c.at < now && Math.hypot(c.cx - cx, c.cy - cy) < Math.max(c.w, b.w) * 0.6);
    this.ySoft.push({ cx, cy, w: b.w, at: now });
    this.yAgainUntil = now + 400;
    return hit;
  }

  // YuNetが顔と判断した枠で、追っている顔を確かめ、位置を合わせる
  _yConfirm(t, b, ok, job, tpl, now) {
    t.yAt = now;
    t.yBest = Math.max(t.yBest || 0, b.s);
    if (t.clear) return;
    if (!t.yRotS || b.s >= t.yRotS - 0.05 || now - t.yRotAt > 1500) { t.yRot = b.rot; t.yRotS = b.s; t.yRotAt = now; }
    if (!ok) return;
    t.yOkAt = now;
    t.fake = false; t.verified = true; t.probation = 0; t.unsureN = 0; t.lowN = 0;
    // 調べている間に、もっと新しい検出で位置が更新されていなければ、位置と大きさも合わせる
    if ((t.upAt || 0) <= job.at || t.rsrc === 'ncc') {
      const a = 0.7;
      // 斜めの向きで見つけた枠は外側を囲む大きさなので、今の大きさより広げない
      const diag = b.rot % 90 !== 0;
      const bw = diag ? Math.min(b.w, t.w) : b.w, bh = diag ? Math.min(b.h, t.h) : b.h;
      t.w += (bw - t.w) * 0.5; t.h += (bh - t.h) * 0.5;
      t.x += (b.x + b.w / 2 - t.w / 2 - t.x) * a; t.y += (b.y + b.h / 2 - t.h / 2 - t.y) * a;
    }
    this._learnLook(t, this.gray, this.gw, this.gh, this.gk, t);
    t.seen = now;
    t.kp = yunetKp(b.kp); t.kpAt = now;
  }

  _applyYunet(now, W, H) {
    if (!this.yQueue.length) return;
    for (const { job, boxes } of this.yQueue.splice(0)) {
      if (job.W !== W || job.H !== H || job.gw !== this.gw) continue;
      const asked = new Set(job.cells.map((c) => c.prop).filter(Boolean));
      for (const b0 of boxes) {
        if (b0.s < Y_HINT || b0.w < 8 || b0.w > W * 1.3) continue;
        // 調べた時の見た目を覚え、いまの画面で探し直して、動いた分だけずらす
        const tpl = this._makeTemplate(job.gray, job.gw, job.gh, job.gk, b0);
        // 調べている間にカメラが動いた分をまず足し、そのあと見た目で細かく合わせる
        const camDx = (this.camX || 0) - job.camX, camDy = (this.camY || 0) - job.camY;
        let dx = camDx, dy = camDy;
        if (tpl) {
          const m = this._matchTemplate(tpl, this.gray, this.gw, this.gh, this.gk, b0.x + b0.w / 2 + camDx, b0.y + b0.h / 2 + camDy, Math.max(b0.w * 0.5, 30));
          if (m && m.ncc >= NCC_OK) { dx = m.cx - (b0.x + b0.w / 2); dy = m.cy - (b0.y + b0.h / 2); }
        }
        const b = { x: b0.x + dx, y: b0.y + dy, w: b0.w, h: b0.h, s: b0.s, rot: b0.rot, kp: b0.kp.map((p) => ({ x: p.x + dx, y: p.y + dy })) };
        if (job.kind === 'roi') {
          // 顔の周りを調べた回：その区画の真ん中あたりで見つかった顔だけを、その顔の手がかりにする
          const c = job.cells[b0.cell], t = c && c.track;
          if (c && c.prop) {
            const d0 = Math.hypot(b0.x + b0.w / 2 - c.cx, b0.y + b0.h / 2 - c.cy);
            if (d0 > c.side * 0.3 || b0.w < c.side * 0.12 || b0.w > c.side * 0.9) continue;
            if (b0.s < (b0.rot ? Y_PROP_OK : Y_PROP_OK_UP) || c.prop.hit) continue;
            c.prop.hit = true;
            const same = this.tracks.find((o) => coverage(o, b) > 0.3);
            if (same) this._yConfirm(same, b, true, job, tpl, now); else this._birthY(b, true, now);
            continue;
          }
          if (!t || !this.tracks.includes(t)) continue;
          const d = Math.hypot(b0.x + b0.w / 2 - c.cx, b0.y + b0.h / 2 - c.cy);
          if (d > c.side * 0.3 || b0.w < c.side * 0.12 || b0.w > c.side * 0.9) continue;
          // まだ顔と確かめられていない枠は、基準を上げる（拡大するとライトなどもそれらしく見えることがあるため）
          const known = t.verified && !t.fake;
          const agree = now - (t.dAt || -1e9) < 300 && b0.s >= Y_AGREE + 0.05;
          const firm = b0.s >= (known ? Y_OK : Y_OK_TURNED);
          // まだ顔と確かめられていない枠は、高い点数か、もとの顔検出も反応している場所を拡大して合格点が出たとき、
          // または続けて反応したときだけ「顔」とする
          const both = firm && now - (t.dAt || -1e9) < 300;
          const pass = known ? firm || agree : b0.s >= Y_SURE || both || ((firm || agree) && this._yTwice(b, now));
          if (pass && !firm) t.weakAt = now; else if (firm) t.weakAt = 0;
          this._yConfirm(t, b, pass, job, tpl, now);
          continue;
        }
        const line = b0.s >= (b0.rot ? Y_OK_TURNED : Y_OK);
        const sure = b0.s >= (b0.rot ? Y_SURE : Y_SURE_UP);
        let best = null, bestScore = 0;
        for (const t of this.tracks) {
          const s = overlapScore(t, b);
          if (s > bestScore) { bestScore = s; best = t; }
        }
        if (best && bestScore > 0.15) {
          // 2つの検出が同じ場所に反応していれば、点数が少し低くても顔とする（動いてぶれた寝顔など）
          const known = best.verified && !best.fake;
          const agree = bestScore > 0.3 && now - (best.dAt || -1e9) < 300 && b0.s >= Y_AGREE;
          const pass = known ? line || agree : sure || ((line || agree) && this._yTwice(b, now));
          if (pass && !line) best.weakAt = now; else if (line) best.weakAt = 0;
          this._yConfirm(best, b, pass, job, tpl, now);
        } else if (!this.tracks.some((t) => coverage(t, b) > 0.3)) {
          if (sure) { this._birthY(b, true, now); continue; }
          // カメラを振った直後に合格点の反応があれば、確かめる間も仮にぼかしておく
          if (line && now - (this.movedAt || -1e9) < Y_HOT_MS) this._birthY(b, this._yTwice(b, now), now);
          else if (b0.s >= Y_PROP) this._propose(b, now);
        }
      }
      for (const p of asked) if (!p.hit) (this.yRejects = this.yRejects || []).push({ cx: p.cx, cy: p.cy, w: p.w, at: now });
    }
  }

  // ぼかす範囲（画面の外にはみ出したまま）
  _regionRaw(t) {
    const side = t.rot === 90 || t.rot === 270;   // 横倒しの顔
    const p = this._predict(t, this.nowTs || 0);
    const cx = p.cx, cy = p.cy - (t.rot ? 0 : t.h * 0.08);
    // 速く動いているときは、遅れを見込んで範囲を広げる
    const speed = Math.hypot(t.vx || 0, t.vy || 0);
    const extra = speed < 0.15 ? 0 : Math.min(speed * 90, t.w * 0.5);   // ゆっくりのときは広げない
    const w = t.w * (side ? EXPAND_H : EXPAND_W) + extra, h = t.h * (side ? EXPAND_W : EXPAND_H) + extra;
    const r = { x: cx - w / 2, y: cy - h / 2, w, h };
    // 画面の端の近くの顔は、ぼかしを画面の端まで広げる。半分はみ出した顔は検出も追いかけも
    // ずれやすく、ぼかしが内側（胸など）へずれて、端に残った顔が見えてしまうのを防ぐ
    const W = this.out.width, H = this.out.height;
    if (W && H) {
      if (r.x > 0 && r.x < r.w * EDGE_REACH) { r.w += r.x; r.x = 0; }
      if (r.x + r.w < W && W - (r.x + r.w) < r.w * EDGE_REACH) r.w = W - r.x;
      if (r.y > 0 && r.y < r.h * EDGE_REACH) { r.h += r.y; r.y = 0; }
      if (r.y + r.h < H && H - (r.y + r.h) < r.h * EDGE_REACH) r.h = H - r.y;
    }
    return r;
  }

  _region(t, W, H) { return clampRect(this._regionRaw(t), W, H); }

  // ---- 映像を少し遅らせて出す ----
  // いまのコマと、その時点の枠の位置をためる。新しく見つかった顔は、ためてあるコマにもさかのぼって印を付ける
  _pushFrame(W, H, now, detected) {
    const q = this.delayQ, cam = { x: this.camX || 0, y: this.camY || 0 };
    const marks = [];
    for (const t of this.tracks) {
      if (t.verified && !t.fake) t.everOk = true;
      const r = this._regionRaw(t);
      marks.push({ t, r });
      if (t.backDone) continue;
      t.backDone = true;
      for (const e of q) {
        // カメラが動いた分だけ位置を戻す
        const w = r.w * BACK_GROW, h = r.h * BACK_GROW;
        e.marks.push({ t, r: { x: r.x + (e.cam.x - cam.x) - (w - r.w) / 2, y: r.y + (e.cam.y - cam.y) - (h - r.h) / 2, w, h } });
      }
    }
    let f = this.framePool.pop();
    if (!f) {
      const canvas = document.createElement('canvas');
      f = { canvas, ctx: canvas.getContext('2d', { alpha: false }) };
    }
    if (f.canvas.width !== W || f.canvas.height !== H) { f.canvas.width = W; f.canvas.height = H; }
    f.ctx.drawImage(this.raw, 0, 0);
    q.push({ f, at: now, cam, marks, blurAll: !detected, W, H });
    while (q.length > OUT_MAX_FRAMES) this.framePool.push(q.shift().f);
  }

  // 遅らせた時間がたったコマを、ぼかしてから録画される画面へ描く
  _drawDelayed(W, H, now) {
    const q = this.delayQ;
    let e = null;
    while (q.length && now - q[0].at >= OUT_DELAY_MS) {
      if (e) this.framePool.push(e.f);
      e = q.shift();
    }
    if (!e) {
      // 最初のコマがたまるまでは、画面全体をぼかしておく
      if (!this.shown) { this._blurAll(W, H, this.raw); this._drawLabel(W, H); }
      return;
    }
    const list = [];
    if (e.blurAll || e.W !== W || e.H !== H) {
      this._blurAll(W, H, e.f.canvas);
    } else {
      this.octx.drawImage(e.f.canvas, 0, 0);
      const done = new Set();
      for (const m of e.marks) {
        let t = m.t;
        while (t.mergedInto) t = t.mergedInto;
        if (done.has(t) || t.fake) continue;
        // 消えた枠のうち、一度も顔と確かめられなかったものは、ぼかさない
        if (!t.everOk && !t.clear && !this.tracks.includes(t)) continue;
        done.add(t);
        const r = clampRect(this._smooth(t, m.r, e.at), W, H);
        if (r.w < 2 || r.h < 2) continue;
        list.push({ t, r });
        if (t.clear) continue;
        try {
          this._blurFace(r, e.f.canvas);
        } catch (err) {
          // ぼかしの描画に失敗したときは、その範囲を塗りつぶして顔が見えないようにする
          this.octx.fillStyle = '#888';
          this.octx.fillRect(r.x, r.y, r.w, r.h);
        }
      }
    }
    this._drawLabel(W, H);
    this.shown = { at: e.at, marks: list };
    // 録画中は「ぼかしなし」にした人の位置を覚えておく（撮影後の探し直しで、その人にぼかしを足さないため）。
    // 位置と時刻だけで、映像は覚えない
    if (this.clearLog && now - (this.clearLogLast || 0) >= 100) {
      const c = list.filter((x) => x.t.clear);
      if (c.length) {
        this.clearLogLast = now;
        this.clearLog.push({ t: (now - this.clearLogAt) / 1000, W, H, r: c.map((x) => ({ x: x.r.x, y: x.r.y, w: x.r.w, h: x.r.h })) });
      }
    }
    this.framePool.push(e.f);
  }

  // 出すコマごとのぼかし範囲を、顔ごとになめらかにする（R はそのコマで覆うべき範囲）
  _smooth(t, R, at) {
    const s = t.sm;
    const cx = R.x + R.w / 2, cy = R.y + R.h / 2;
    if (!s || at <= s.at || at - s.at > SM_RESET_MS) {
      t.sm = { cx, cy, w: R.w, h: R.h, at };
      return R;
    }
    const dt = at - s.at;
    const d = Math.hypot(cx - s.cx, cy - s.cy) / Math.max(1, Math.min(R.w, R.h));
    const a = d >= SM_SNAP ? 1 : 1 - Math.exp(-dt / (SM_POS_MS * (1 - d / SM_SNAP) + 1));
    const nx = s.cx + (cx - s.cx) * a, ny = s.cy + (cy - s.cy) * a;
    // ならした中心から、元の範囲を覆いきる大きさ
    const needW = R.w + 2 * Math.abs(cx - nx), needH = R.h + 2 * Math.abs(cy - ny);
    const k = Math.exp(-dt / SM_SHRINK_MS);
    const w = needW >= s.w ? needW : needW + (s.w - needW) * k;
    const h = needH >= s.h ? needH : needH + (s.h - needH) * k;
    t.sm = { cx: nx, cy: ny, w, h, at };
    return { x: nx - w / 2, y: ny - h / 2, w, h };
  }

  _flushDelay() {
    for (const e of this.delayQ) this.framePool.push(e.f);
    this.delayQ = [];
    this.shown = null;
  }

  // すりガラスのような、なめらかなぼかし。
  // 手順：顔の範囲を段階的に小さくして細部を消す → 少し大きくして、ずらし重ねでならす
  //       → ふちをやわらかく消す → 元の大きさに戻して重ねる
  _blurFace(r, src) {
    const ratio = r.h / r.w;
    const size = (c, w) => {
      const h = Math.max(2, Math.round(w * ratio));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      return h;
    };
    // 1) 顔の範囲を小さな画面に写す（段階的に縮小。一気に縮めると映像がちらつくため）
    const MW = 72;
    const ph = size(this.pre, MW * 2);
    this.pctx.imageSmoothingEnabled = true;
    this.pctx.drawImage(src, r.x, r.y, r.w, r.h, 0, 0, this.pre.width, ph);
    const mh = size(this.midA, MW); size(this.midB, MW);
    const a = this.actx, b = this.bctx;
    a.imageSmoothingEnabled = true;
    a.globalAlpha = 1;
    a.drawImage(this.pre, 0, 0, this.pre.width, ph, 0, 0, MW, mh);
    // 2) にじませる：少しずつずらした絵を重ねて平均する。幅を変えて3回くり返すと、
    //    ます目の跡が残らない、こすったようなぼかしになる
    b.globalCompositeOperation = 'source-over';
    const smear = (dst, src, d) => {
      dst.globalAlpha = 1;
      dst.drawImage(src, 0, 0);
      let n = 2;
      for (let oy = -d; oy <= d; oy += d) for (let ox = -d; ox <= d; ox += d) {
        if (!ox && !oy) continue;
        dst.globalAlpha = 1 / n++;
        dst.drawImage(src, ox, oy);
      }
      dst.globalAlpha = 1;
    };
    smear(b, this.midA, BLUR_STEPS[0]);
    smear(a, this.midB, BLUR_STEPS[1]);
    smear(b, this.midA, BLUR_STEPS[2]);
    // 3) うっすら白くして、曇りガラスのように明るくする
    b.fillStyle = `rgba(255,255,255,${BLUR_HAZE})`;
    b.fillRect(0, 0, MW, mh);
    // 4) ふちをやわらかく（だ円の外側ほど透明に）
    b.globalCompositeOperation = 'destination-in';
    b.setTransform(MW / 2, 0, 0, mh / 2, MW / 2, mh / 2);
    b.fillStyle = this._mask(b);
    b.fillRect(-1, -1, 2, 2);
    b.setTransform(1, 0, 0, 1, 0, 0);
    b.globalCompositeOperation = 'source-over';
    // 5) 録画される画面へ
    const o = this.octx;
    o.imageSmoothingEnabled = true;
    o.drawImage(this.midB, 0, 0, MW, mh, r.x, r.y, r.w, r.h);
  }

  _mask(ctx) {
    if (!this._maskGrad) {
      const g = ctx.createRadialGradient(0, 0, 0.78, 0, 0, 1);
      g.addColorStop(0, 'rgba(0,0,0,1)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      this._maskGrad = g;
    }
    return this._maskGrad;
  }

  _blurAll(W, H, src) {
    const sw = 16, sh = Math.max(2, Math.round(16 * H / W));
    if (this.small.width !== sw || this.small.height !== sh) { this.small.width = sw; this.small.height = sh; }
    this.sctx.drawImage(src, 0, 0, W, H, 0, 0, sw, sh);
    this.octx.imageSmoothingEnabled = true;
    this.octx.drawImage(this.small, 0, 0, sw, sh, 0, 0, W, H);
  }

  _drawLabel(W, H) {
    if (!this.label) return;
    const o = this.octx;
    const fs = Math.round(Math.min(W, H) * 0.042);
    o.save();
    o.font = `700 ${fs}px -apple-system, "Hiragino Sans", "Yu Gothic UI", Meiryo, sans-serif`;
    o.textBaseline = 'middle';
    const pad = fs * 0.5;
    const tw = o.measureText(this.label).width;
    const x = fs * 0.6, y = H - fs * 0.6 - fs * 1.6;
    o.fillStyle = 'rgba(0,0,0,0.6)';
    o.fillRect(x, y, tw + pad * 2, fs * 1.6);
    o.fillStyle = '#fff';
    o.fillText(this.label, x + pad, y + fs * 0.82);
    o.restore();
  }

  _drawOverlay(W, H) {
    const c = this.vctx;
    c.clearRect(0, 0, W, H);
    const lw = Math.max(3, Math.round(Math.min(W, H) * 0.006));
    for (const { t, r } of this.shown ? this.shown.marks : []) {
      c.lineWidth = lw;
      if (t.clear) {
        c.strokeStyle = '#34d399';
        c.setLineDash([]);
      } else {
        c.strokeStyle = 'rgba(255,255,255,0.75)';
        c.setLineDash([lw * 3, lw * 2]);
      }
      c.beginPath();
      c.ellipse(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2, 0, 0, Math.PI * 2);
      c.stroke();
      if (t.clear) {
        const fs = Math.round(Math.min(W, H) * 0.032);
        c.font = `700 ${fs}px sans-serif`;
        c.textAlign = 'center';
        c.fillStyle = '#34d399';
        c.fillText('ぼかしなし', r.x + r.w / 2, Math.max(fs, r.y - fs * 0.4));
      }
    }
    c.setLineDash([]);
  }

  // ---- タップで切り替え ----
  _onTap(e) {
    const W = this.out.width, H = this.out.height;
    if (!W || !H) return;
    const rect = this.out.getBoundingClientRect();
    const s = Math.min(rect.width / W, rect.height / H);
    const offX = (rect.width - W * s) / 2, offY = (rect.height - H * s) / 2;
    const px = (e.clientX - rect.left - offX) / s;
    const py = (e.clientY - rect.top - offY) / s;
    const shownMarks = this.shown ? this.shown.marks : [];
    let hit = null, hitArea = Infinity;
    for (const { t, r } of shownMarks) {
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h && r.w * r.h < hitArea) {
        hit = t; hitArea = r.w * r.h;
      }
    }
    if (!hit) {
      // 枠のすぐ外を押したときは、いちばん近い枠を選ぶ
      let best = Infinity;
      for (const { t, r } of shownMarks) {
        const d = Math.hypot((px - r.x - r.w / 2) / (r.w / 2), (py - r.y - r.h / 2) / (r.h / 2));
        if (d < 1.35 && d < best) { best = d; hit = t; }
      }
    }
    if (hit) {
      hit.clear = !hit.clear;
      hit.hits = 0; hit.misses = 0; hit.nextId = 0;
      if (hit.clear) {
        hit.verified = true;
        // この人の顔を覚える（次の照合のタイミングで特徴をとる）
        hit.noAuto = false;
        hit.person = { id: this.nextPerson++, samples: [] };
        this.persons.push(hit.person);
      } else {
        // ぼかしに戻した人は、覚えた特徴を消す。この顔は自動では外さない
        if (hit.person) this.persons = this.persons.filter((p) => p !== hit.person);
        hit.person = null;
        hit.noAuto = true;
      }
      if (this.onToggle) this.onToggle(hit.clear);
    }
  }
}

// 体の点から頭の位置を求める。顔の点と両肩がはっきり見えているときだけ採用する（足だけが映った場面の誤認を防ぐ）
function poseHead(lm, W, H) {
  if (!lm || lm.length < 25) return null;
  const P = lm.map((p) => ({ x: p.x * W, y: p.y * H, v: p.v }));
  const face = P.slice(0, 11).filter((p) => p.v > POSE_VIS);
  const sa = P[11], sb = P[12];
  if (face.length < 4 || sa.v < POSE_VIS || sb.v < POSE_VIS) return null;
  const inside = (p) => p.x > -10 && p.x < W + 10 && p.y > -10 && p.y < H + 10;
  if (!inside(sa) || !inside(sb)) return null;
  const cx = face.reduce((a, p) => a + p.x, 0) / face.length, cy = face.reduce((a, p) => a + p.y, 0) / face.length;
  // 腰が見えているときは、頭が肩より腰の反対側にあることを確かめる（胸を頭と取りちがえるのを防ぐ）
  const ha = P[23], hb = P[24];
  if (ha.v > POSE_VIS && hb.v > POSE_VIS) {
    const mx = (sa.x + sb.x) / 2, my = (sa.y + sb.y) / 2, ux = mx - (ha.x + hb.x) / 2, uy = my - (ha.y + hb.y) / 2;
    if ((cx - mx) * ux + (cy - my) * uy <= 0) return null;
  }
  const d = (a, b) => Math.hypot(P[a].x - P[b].x, P[a].y - P[b].y);
  const r = Math.max(d(7, 8) * 0.62, d(0, 9) * 1.5, d(11, 12) * 0.28, 12);
  return { cx, cy, r };
}

// 枠を画面の中に収める
function clampRect(r, W, H) {
  const x0 = Math.max(0, Math.floor(r.x)), y0 = Math.max(0, Math.floor(r.y));
  const x1 = Math.min(W, Math.ceil(r.x + r.w)), y1 = Math.min(H, Math.ceil(r.y + r.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// 2つの枠が同じ顔を指しているか（小さいほうの大部分が重なり、中心も近い）
function sameFace(a, b) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y), x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const ix = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  if (ix <= 0) return false;
  const small = Math.min(a.w * a.h, b.w * b.h);
  const d = Math.hypot(a.x + a.w / 2 - b.x - b.w / 2, a.y + a.h / 2 - b.y - b.h / 2);
  return ix / small > 0.45 && d < Math.max(a.w, b.w) * 0.6;
}

// drop の情報を keep に引き継ぐ
function absorbTrack(keep, drop) {
  for (const k of ['seen', 'yAt', 'yOkAt', 'dAt', 'followAt', 'fullAt', 'poseAt']) keep[k] = Math.max(keep[k] || 0, drop[k] || 0);
  keep.strong = keep.strong || drop.strong;
  if (!drop.fake) keep.verified = keep.verified || drop.verified;
  if (drop.kpAt > (keep.kpAt || 0)) { keep.kp = drop.kp; keep.kpAt = drop.kpAt; }
  // 大きすぎる枠と、顔に合った枠が重なったときは、顔に合った大きさに近づける
  if (!drop.fake && drop.w < keep.w * 0.85 && drop.w > keep.w * 0.5) {
    const cx = (keep.x + keep.w / 2 + drop.x + drop.w / 2) / 2, cy = (keep.y + keep.h / 2 + drop.y + drop.h / 2) / 2;
    keep.w = Math.min(keep.w, drop.w * 1.25); keep.h = Math.min(keep.h, drop.h * 1.25);
    keep.x = cx - keep.w / 2; keep.y = cy - keep.h / 2;
    keep.tpl = null;
  }
}

// YuNetの顔の点（右目・左目・鼻・口の右端・口の左端）を、顔の見分けで使う並び（右目・左目・鼻・口）にする
function yunetKp(p) {
  if (!p || p.length < 5) return [];
  return [p[0], p[1], p[2], { x: (p[3].x + p[4].x) / 2, y: (p[3].y + p[4].y) / 2 }];
}

function nearest(person, desc) {
  let min = Infinity;
  for (const s of person.samples) { const d = distance(s, desc); if (d < min) min = d; }
  return min;
}

// 2つの枠の重なりが、小さいほうの枠に占める割合
function coverage(t, b) {
  const x0 = Math.max(t.x, b.x), y0 = Math.max(t.y, b.y);
  const x1 = Math.min(t.x + t.w, b.x + b.w), y1 = Math.min(t.y + t.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (Math.min(t.w * t.h, b.w * b.h) || 1);
}

function overlapScore(t, b) {
  const x0 = Math.max(t.x, b.x), y0 = Math.max(t.y, b.y);
  const x1 = Math.min(t.x + t.w, b.x + b.w), y1 = Math.min(t.y + t.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const iou = inter / (t.w * t.h + b.w * b.h - inter || 1);
  // 速く動いて重なりが小さくても、中心が近ければ同じ顔とみなす
  const dx = (t.x + t.w / 2) - (b.x + b.w / 2), dy = (t.y + t.h / 2) - (b.y + b.h / 2);
  const dist = Math.hypot(dx, dy) / Math.max(t.w, b.w, 1);
  const near = dist < 0.7 ? (0.7 - dist) / 0.7 * 0.5 : 0;
  return Math.max(iou, near);
}
