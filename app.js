// 画面の切り替えと、タイトル一覧の管理
import { BlurCamera } from './camera.js';
import { CanvasRecorder, buildFileName, saveVideo } from './recorder.js';

const $ = (id) => document.getElementById(id);
import { RedactEditor, exportRedacted } from './redact.js';

// 場面ごとの、最初から入っている項目。タブごとに並べ替え・削除・追加ができる
const SETS = [
  { id: 'family', name: '家族指導', items: ['吸引', 'オムツ交換', '胃ろう注入', '着替え', '車いす移乗', '体位変換', '食事介助', '口腔ケア', '服薬介助'] },
  { id: 'rehab', name: 'リハビリ', items: ['歩行', '立ち上がり', '車いす移乗', '自主トレ', '嚥下訓練', '関節可動域訓練', '階段昇降'] },
  { id: 'home', name: '訪問ケア', items: ['申し送り', '吸引', '胃ろう注入', 'オムツ交換', '入浴介助', '褥瘡ケア', '服薬介助'] },
  { id: 'staff', name: '院内・教育', items: ['申し送り', '手技練習', '体位変換', '吸引', '車いす移乗', '口腔ケア'] },
];
const KEY_SETS = 'kaigoCamera.titleSets';
const KEY_TAB = 'kaigoCamera.titleTab';
const KEY_OLD = 'kaigoCamera.customTitles';   // 前の版で追加した項目

const screens = { title: $('screenTitle'), camera: $('screenCamera'), review: $('screenReview') };
function show(name) {
  for (const [k, el] of Object.entries(screens)) el.hidden = k !== name;
  document.documentElement.classList.toggle('camOn', name === 'camera');
  window.scrollTo(0, 0);
}

// ---- 画面1：内容を選ぶ ----
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 保存できなくても動かす */ } },
};
// タブごとの項目の並び（端末の中に保存）
let lists = {};
try { lists = JSON.parse(store.get(KEY_SETS) || '{}') || {}; } catch (e) { lists = {}; }
for (const s of SETS) if (!Array.isArray(lists[s.id])) lists[s.id] = [...s.items];
// 前の版で追加した項目は「家族指導」に引き継ぐ
try {
  const old = JSON.parse(store.get(KEY_OLD) || '[]');
  if (Array.isArray(old) && old.length && !store.get(KEY_SETS)) {
    for (const n of old) if (typeof n === 'string' && !lists.family.includes(n)) lists.family.push(n);
  }
} catch (e) { /* 何もしない */ }
let tab = store.get(KEY_TAB);
if (!SETS.some((s) => s.id === tab)) tab = SETS[0].id;
let selected = '';
let editing = false;

function saveLists() { store.set(KEY_SETS, JSON.stringify(lists)); }

function renderTitles() {
  // タブ
  const tabs = $('setTabs');
  tabs.textContent = '';
  for (const s of SETS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tab' + (s.id === tab ? ' on' : '');
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', String(s.id === tab));
    b.textContent = s.name;
    b.addEventListener('click', () => { tab = s.id; store.set(KEY_TAB, tab); renderTitles(); });
    tabs.appendChild(b);
  }
  const items = lists[tab];
  // 項目（選ぶ）
  const list = $('titleList');
  list.textContent = '';
  list.hidden = editing;
  for (const name of items) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip' + (name === selected ? ' on' : '');
    b.textContent = name;
    b.addEventListener('click', () => { selected = name; renderTitles(); });
    list.appendChild(b);
  }
  if (!items.length) {
    const p = document.createElement('p');
    p.className = 'note';
    p.textContent = '項目がありません。下の欄から追加してください。';
    list.appendChild(p);
  }
  // 並べ替え・削除
  const ed = $('editList');
  ed.textContent = '';
  ed.hidden = !editing;
  if (editing) {
    items.forEach((name, i) => {
      const row = document.createElement('div');
      row.className = 'editRow';
      const label = document.createElement('span');
      label.textContent = name;
      row.appendChild(label);
      const mk = (text, aria, fn, disabled, cls) => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'mini' + (cls ? ' ' + cls : ''); b.textContent = text;
        b.setAttribute('aria-label', aria); b.disabled = disabled;
        b.addEventListener('click', fn);
        row.appendChild(b);
      };
      const move = (d) => () => { const a = lists[tab]; [a[i], a[i + d]] = [a[i + d], a[i]]; saveLists(); renderTitles(); };
      mk('▲', `${name}を上へ`, move(-1), i === 0);
      mk('▼', `${name}を下へ`, move(1), i === items.length - 1);
      mk('削除', `${name}を削除`, () => {
        lists[tab] = lists[tab].filter((n) => n !== name);
        if (selected === name && !SETS.some((s) => lists[s.id].includes(name))) selected = '';
        saveLists(); renderTitles();
      }, false, 'del');
      ed.appendChild(row);
    });
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'linkBtn';
    reset.textContent = 'このタブを最初の状態に戻す';
    reset.addEventListener('click', () => {
      const s = SETS.find((x) => x.id === tab);
      if (!confirm(`「${s.name}」の項目を最初の状態に戻します。追加した項目と並び順は消えます。よろしいですか？`)) return;
      lists[tab] = [...s.items];
      saveLists(); renderTitles();
    });
    ed.appendChild(reset);
  }
  $('editTitles').textContent = editing ? '並べ替えを終える' : '並べ替え・削除';
  $('addForm').hidden = editing;
  $('titleNote').textContent = editing
    ? '▲▼で順番を入れ替え、「削除」でこのタブから外せます。'
    : '追加した項目は、いま開いているタブに入ります。';
  const start = $('startCamera');
  start.disabled = !selected || editing;
  start.textContent = editing ? '並べ替えを終えてください' : selected ? `「${selected}」を撮影する` : '内容を選んでください';
}

$('editTitles').addEventListener('click', () => { editing = !editing; renderTitles(); });

$('addForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const v = $('addInput').value.trim();
  if (!v) return;
  if (!lists[tab].includes(v)) { lists[tab].push(v); saveLists(); }
  selected = v;
  $('addInput').value = '';
  renderTitles();
});

// ---- 画面2：カメラ ----
const camera = new BlurCamera({ video: $('cam'), out: $('out'), overlay: $('overlay') });
const recorder = new CanvasRecorder($('out'));
window.__kaigoCamera = camera; // 動作確認用
window.__kaigoRecorder = recorder;
// 一度タップしたら案内文を消す（「ぼかしなし」の表示と重ならないように）
camera.onToggle = () => { $('hint').hidden = true; };
let wakeLock = null;
let timerId = null;
let result = null; // { blob, url, fileName }

function setStatus(text) {
  $('status').hidden = !text;
  $('status').textContent = text || '';
}

function labelText() {
  const d = new Date();
  return `${selected}　${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()}`;
}

$('startCamera').addEventListener('click', async () => {
  if (!selected) return;
  show('camera');
  $('recBtn').disabled = true;
  $('hint').hidden = false;
  camera.setLabel(labelText());
  camera.searchTurned = $('lyingToggle').checked;
  setStatus('カメラと顔検出を準備しています…\n（準備ができるまで画面全体がぼけます）');
  try {
    await camera.start();
    setStatus('');
    $('recBtn').disabled = false;
  } catch (e) {
    const denied = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
    setStatus(denied
      ? 'カメラの使用が許可されていません。\nSafariの設定でカメラを許可してから、もう一度開いてください。'
      : '準備に失敗しました。\n' + (e && e.message ? e.message : e) + '\n「もどる」を押してやり直してください。');
  }
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (e) { wakeLock = null; }
});

function leaveCamera() {
  camera.lock(false);
  camera.stop();
  clearInterval(timerId);
  $('recBadge').hidden = true;
  $('recBtn').classList.remove('on');
  try { wakeLock?.release(); } catch (e) { /* 何もしない */ }
  wakeLock = null;
}

$('backBtn').addEventListener('click', async () => {
  if (recorder.recording) {
    // 録画中に戻る＝破棄
    try { await recorder.stop(); } catch (e) { /* 何もしない */ }
  }
  leaveCamera();
  show('title');
});

// 録画中でも前後のカメラを切り替えられる（切り替えた直後は、少しの間画面全体がぼけます）
$('flipBtn').addEventListener('click', async () => {
  const btn = $('flipBtn');
  if (btn.disabled) return;
  btn.disabled = true;
  try {
    await camera.flip();
  } catch (e) {
    flash('カメラを切り替えられませんでした。', 2000);
  } finally {
    btn.disabled = false;
  }
  // カメラを開き直すと、マイクが止まる端末がある。そのときは知らせる
  if (recorder.recording && micOn && !recorder.micLive) flash('マイクが止まりました。ここから先は音声が入りません。', 4000);
});

// ---- 音声あり／なし ----
let micOn = $('audioToggle').checked;
function flash(text, ms) {
  setStatus(text);
  clearTimeout(flash.timer);
  flash.timer = setTimeout(() => setStatus(''), ms);
}
function renderMic() {
  const b = $('micBtn');
  b.classList.toggle('off', !micOn);
  b.setAttribute('aria-pressed', String(micOn));
  b.setAttribute('aria-label', micOn ? '音声あり（押すと音声なし）' : '音声なし（押すと音声あり）');
  $('micText').textContent = micOn ? '音声あり' : '音声なし';
}
$('micBtn').addEventListener('click', () => {
  micOn = !micOn;
  $('audioToggle').checked = micOn;
  renderMic();
  if (recorder.recording) {
    const ok = recorder.setAudio(micOn);
    if (micOn && !ok) flash('マイクが使えないため、音声は入りません。', 3000);
    else flash(micOn ? 'ここから音声を録音します' : 'ここから音声なし（無音）で録画します', 1800);
  }
});
$('audioToggle').addEventListener('change', () => { micOn = $('audioToggle').checked; renderMic(); });
renderMic();

$('recBtn').addEventListener('click', async () => {
  const btn = $('recBtn');
  btn.disabled = true;
  try {
    if (!recorder.recording) {
      if (!camera.ready) throw new Error('顔検出の準備ができていません');
      camera.setLabel(labelText());
      camera.lock(true);
      const { audioOk } = await recorder.start({ audio: micOn });
      btn.classList.add('on');
      btn.setAttribute('aria-label', '録画停止');
      $('recBadge').hidden = false;
      $('recTime').textContent = '0:00';
      timerId = setInterval(() => {
        const s = Math.floor(recorder.elapsedMs / 1000);
        $('recTime').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
      }, 500);
      if (micOn && !audioOk) flash('マイクが使えないため、音声なしで録画しています。', 3000);
      // 録画中に気が変わっても、押した時点の設定にそろえる
      recorder.setAudio(micOn);
    } else {
      const startedDate = new Date(Date.now() - recorder.elapsedMs);
      const { blob, type, durationMs } = await recorder.stop();
      btn.setAttribute('aria-label', '録画開始');
      leaveCamera();
      openReview(blob, type, startedDate, durationMs);
    }
  } catch (e) {
    camera.lock(false);
    setStatus('録画できませんでした。\n' + (e && e.message ? e.message : e));
    setTimeout(() => setStatus(''), 4000);
  } finally {
    btn.disabled = false;
  }
});

// アプリを閉じたり別の画面に切り替えたりしたら、録画中の動画は破棄する（途中までの動画を残さない）
document.addEventListener('visibilitychange', async () => {
  if (document.hidden && recorder.recording) {
    try { await recorder.stop(); } catch (e) { /* 何もしない */ }
    leaveCamera();
    show('title');
  }
  // 手動ぼかしの作り直し中に画面が消えると、再生が止まって途中までの動画になるので中止する
  if (document.hidden && exportJob) {
    exportJob.cancel();
    exportLost = true;
  }
});

// ---- 画面3：確認 ----
// result.orig：撮ったままの動画（自動ぼかし済み）。result.blob：保存する動画（手動ぼかしを入れたらその動画）
function openReview(blob, type, date, durationMs) {
  const fileName = buildFileName(selected, date, type);
  const url = URL.createObjectURL(blob);
  result = { orig: blob, origUrl: url, blob, url, fileName, type, durationMs };
  editor.reset();
  $('reviewName').textContent = fileName;
  setReviewSrc(url);
  renderInfo();
  $('saveMsg').textContent = '';
  $('againBtn').hidden = true;
  $('saveBtn').disabled = blob.size === 0;
  $('maskStart').disabled = blob.size === 0;
  $('maskDoneNote').hidden = true;
  setMaskMode('off');
  if (blob.size === 0) $('saveMsg').textContent = '動画が空でした。この端末では録画がうまくできていません。';
  show('review');
}

function renderInfo() {
  const mb = (result.blob.size / 1024 / 1024).toFixed(1);
  const s = Math.round(result.durationMs / 1000);
  $('reviewInfo').textContent = `長さ 約${Math.floor(s / 60)}分${s % 60}秒 ／ ${mb} MB ／ 形式 ${result.type}`;
}

function setReviewSrc(url) {
  const v = $('review');
  if (v.getAttribute('src') !== url) v.src = url;
}

function dropResult() {
  if (!result) return;
  if (exportJob) exportJob.cancel();
  const v = $('review');
  v.pause(); v.removeAttribute('src'); v.load();
  URL.revokeObjectURL(result.origUrl);
  if (result.url !== result.origUrl) URL.revokeObjectURL(result.url);
  result = null;
  editor.reset();
  editor.close();
}

// ---- 手動ぼかし ----
const editor = new RedactEditor({ video: $('review'), canvas: $('maskCanvas'), onChange: renderMaskUi });
window.__kaigoEditor = editor; // 動作確認用
let exportJob = null;
let exportLost = false;

const fmt = (t) => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`; };
function durSec() {
  const d = $('review').duration;
  return Number.isFinite(d) && d > 0 ? d : (result ? result.durationMs / 1000 : 0);
}

// off：ふつうの確認 ／ edit：囲みを作る ／ busy：作り直し中
function setMaskMode(mode) {
  const v = $('review');
  $('maskStart').hidden = mode !== 'off';
  $('maskTools').hidden = mode !== 'edit';
  $('maskProgress').hidden = mode !== 'busy';
  $('maskCanvas').hidden = mode !== 'edit';
  $('reviewBottom').hidden = mode !== 'off';
  $('saveMsg').hidden = mode !== 'off';
  $('reviewWrap').classList.toggle('editing', mode === 'edit');
  v.controls = mode === 'off';
  if (mode !== 'edit') editor.close();
}

$('maskStart').addEventListener('click', async () => {
  if (!result) return;
  const v = $('review');
  v.pause();
  // 囲むのは、いつも撮ったままの動画の上で行う（作り直すときも、撮ったままの動画から作る）
  if (v.getAttribute('src') !== result.origUrl) {
    const t = v.currentTime;
    setReviewSrc(result.origUrl);
    await new Promise((r) => { v.addEventListener('loadeddata', r, { once: true }); setTimeout(r, 3000); });
    try { v.currentTime = t; } catch (e) { /* 何もしない */ }
  }
  if (!v.videoWidth) await new Promise((r) => { v.addEventListener('loadedmetadata', r, { once: true }); setTimeout(r, 3000); });
  $('maskDoneNote').hidden = true;
  setMaskMode('edit');
  editor.open(durSec());
  renderMaskUi();
});

function renderMaskUi() {
  const ms = editor.masks, sel = editor.sel;
  const list = $('maskList');
  list.textContent = '';
  for (const m of ms) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip small' + (m === sel ? ' on' : '');
    b.textContent = `囲み${m.no}（${m.kind === 'black' ? '黒塗り' : 'ぼかし'}）`;
    b.addEventListener('click', () => {
      // 選んだ囲みが見える場面へ移る
      const v = $('review'), t = v.currentTime;
      if (t < m.from || t > m.to) v.currentTime = m.keys[0].t >= m.from && m.keys[0].t <= m.to ? m.keys[0].t : m.from;
      editor.select(m.id);
    });
    list.appendChild(b);
  }
  $('maskSel').hidden = !sel;
  if (sel) {
    $('maskSelName').textContent = `囲み${sel.no}`;
    const all = sel.from <= 0.05 && sel.to >= durSec() - 0.05;
    $('maskRange').textContent = all ? '最初から最後まで隠す' : `${fmt(sel.from)} 〜 ${fmt(sel.to)} の間だけ隠す`;
    for (const b of document.querySelectorAll('#maskSel [data-kind]')) b.classList.toggle('on', b.dataset.kind === sel.kind);
    for (const b of document.querySelectorAll('#maskSel [data-shape]')) b.classList.toggle('on', b.dataset.shape === (sel.shape || 'rect'));
  }
  $('maskApply').disabled = !result || (!ms.length && result.blob === result.orig);
  $('maskApply').textContent = !ms.length && result && result.blob !== result.orig ? '手動ぼかしをすべて外す' : '隠して作り直す';
}

for (const b of document.querySelectorAll('#maskSel [data-kind]')) b.addEventListener('click', () => editor.setKind(b.dataset.kind));
for (const b of document.querySelectorAll('#maskSel [data-shape]')) b.addEventListener('click', () => editor.setShape(b.dataset.shape));
$('maskAdd').addEventListener('click', () => editor.addOval());
$('maskFrom').addEventListener('click', () => editor.setFrom());
$('maskTo').addEventListener('click', () => editor.setTo());
$('maskAll').addEventListener('click', () => editor.setAll());
$('maskDel').addEventListener('click', () => editor.removeSel());

// 場面を選ぶ
$('maskPlay').addEventListener('click', () => { const v = $('review'); if (v.paused) v.play().catch(() => {}); else v.pause(); });
$('maskSeek').addEventListener('input', (e) => {
  const v = $('review');
  v.pause();
  v.currentTime = (e.target.value / 1000) * durSec();
});
{
  const v = $('review');
  const sync = () => {
    const d = durSec();
    $('maskTime').textContent = fmt(v.currentTime);
    if (d && document.activeElement !== $('maskSeek')) $('maskSeek').value = String(Math.round((v.currentTime / d) * 1000));
  };
  for (const ev of ['timeupdate', 'seeked', 'loadedmetadata']) v.addEventListener(ev, sync);
  v.addEventListener('play', () => { $('maskPlay').textContent = '一時停止'; });
  v.addEventListener('pause', () => { $('maskPlay').textContent = '再生'; });
}

$('maskCancel').addEventListener('click', () => {
  const v = $('review');
  v.pause();
  setMaskMode('off');
  setReviewSrc(result.url);
});

// 隠して作り直す（押したその場で書き出しを始める。iPhoneでは押した直後でないと再生と音の準備が許されない）
$('maskApply').addEventListener('click', () => {
  if (!result) return;
  const v = $('review');
  v.pause();
  // 囲みが1つもなければ、撮ったままの動画に戻す
  if (!editor.masks.length) {
    if (result.url !== result.origUrl) URL.revokeObjectURL(result.url);
    result.blob = result.orig; result.url = result.origUrl;
    setMaskMode('off');
    setReviewSrc(result.url);
    renderInfo();
    $('maskDoneNote').hidden = true;
    return;
  }
  const masks = editor.masks.map((m) => ({ ...m, keys: m.keys.map((k) => ({ ...k })) }));
  exportLost = false;
  $('maskBar').value = 0;
  $('maskPct').textContent = '0%';
  const job = exportRedacted(result.orig, masks, {
    durationSec: result.durationMs / 1000,
    onProgress: (p) => { $('maskBar').value = p; $('maskPct').textContent = `${Math.round(p * 100)}%`; },
  });
  exportJob = job;
  setMaskMode('busy');
  let lock = null;
  navigator.wakeLock?.request('screen').then((l) => { lock = l; }).catch(() => {});
  job.done.then(({ blob, type }) => {
    if (!result || exportJob !== job) return;
    if (!blob.size) throw new Error('empty');
    if (result.url !== result.origUrl) URL.revokeObjectURL(result.url);
    result.blob = blob; result.type = type; result.url = URL.createObjectURL(blob);
    setMaskMode('off');
    setReviewSrc(result.url);
    renderInfo();
    const n = masks.length;
    $('maskDoneNote').textContent = `手動ぼかしを${n}か所入れました。もう一度再生して、隠れているか確かめてください。直すときは、もう一度「名札などを手で隠す」を押します。`;
    $('maskDoneNote').hidden = false;
    $('saveMsg').textContent = '';
  }).catch(() => {
    if (!result || exportJob !== job) return;
    // 中止・失敗：囲みはそのまま残して、作る画面に戻る
    setReviewSrc(result.origUrl);
    setMaskMode('edit');
    editor.open(durSec());
    renderMaskUi();
    flashReview(exportLost
      ? '画面が消えたため、作り直しを中止しました。画面をつけたまま、もう一度「隠して作り直す」を押してください。'
      : '作り直しを中止しました。');
  }).finally(() => {
    if (exportJob === job) exportJob = null;
    try { lock?.release(); } catch (e) { /* 何もしない */ }
  });
});
$('maskStop').addEventListener('click', () => { if (exportJob) exportJob.cancel(); });

function flashReview(text) {
  const n = $('maskDoneNote');
  n.textContent = text;
  n.hidden = false;
}

$('discardBtn').addEventListener('click', () => { dropResult(); show('title'); });
$('againBtn').addEventListener('click', () => { dropResult(); show('title'); });

$('saveBtn').addEventListener('click', async () => {
  if (!result) return;
  const how = await saveVideo(result.blob, result.fileName);
  if (how === 'shared') {
    $('saveMsg').textContent = '共有メニューで選んだ場所に保存されました。';
    $('againBtn').hidden = false;
  } else if (how === 'downloaded') {
    $('saveMsg').textContent = 'ダウンロードフォルダに保存しました。';
    $('againBtn').hidden = false;
  } else {
    $('saveMsg').textContent = '保存をやめました。もう一度「保存する」を押せます。';
  }
});

// ---- 通信がなくても起動できるようにする ----
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').then(() => navigator.serviceWorker.ready).then(() => {
    const n = $('offlineNote');
    n.textContent = '準備完了：次からは、電波のない場所でも起動できます';
    n.hidden = false;
  }).catch(() => { /* 使えない端末では、今までどおり通信して開く */ });
}

renderTitles();
show('title');
