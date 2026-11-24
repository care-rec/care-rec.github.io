// 録画した mp4 に「動画の長さ」を書き込む。
// iPhone の録画（MediaRecorder）は、短いかたまり（fragment）をつないだ mp4 を作るが、ファイルの頭にある
// 長さの欄が 0 のまま。そのため iPhone の再生部品が長さを読み違え（例：1分6秒の動画を28秒と思う）、
// 手動ぼかしの書き出しや探し直しが途中で終わってしまう。
// そこで、かたまりごとの長さを足し合わせて本当の長さを求め、頭の欄（mvhd・tkhd・mdhd）に書き、
// かたまりつなぎの全体の長さ（mehd）も加える。映像そのものは変えない。
// 形が想定と違うときは、何もせず元のまま返す。

const u32 = (v, o) => v.getUint32(o);
const type = (v, o) => String.fromCharCode(v.getUint8(o + 4), v.getUint8(o + 5), v.getUint8(o + 6), v.getUint8(o + 7));

function children(v, start, end) {
  const out = [];
  let o = start;
  while (o + 8 <= end) {
    let size = u32(v, o), hdr = 8;
    if (size === 1) { size = Number(v.getBigUint64(o + 8)); hdr = 16; }
    if (size === 0) size = end - o;
    if (size < hdr || o + size > end) break;
    out.push({ t: type(v, o), o, size, hdr });
    o += size;
  }
  return out;
}
const find = (list, t) => list.find((b) => b.t === t);

// expectedSec：録画した時間（アプリで測ったもの）。計算した長さがこれと大きく違うときは、こちらを使う
export async function fixMp4Duration(blob, expectedSec = 0) {
  try {
    const buf = await blob.arrayBuffer();
    const v = new DataView(buf);
    const top = children(v, 0, buf.byteLength);
    const moov = find(top, 'moov');
    if (!moov || !top.some((b) => b.t === 'moof')) return blob;   // かたまりつなぎでなければ直さない
    const mk = children(v, moov.o + 8, moov.o + moov.size);
    const mvhd = find(mk, 'mvhd'), mvex = find(mk, 'mvex');
    if (!mvhd || !mvex) return blob;
    const mvV = v.getUint8(mvhd.o + 8);
    const movieScale = u32(v, mvhd.o + 8 + (mvV === 1 ? 20 : 12));
    if (!movieScale) return blob;
    // トラックごとの情報（時間の単位・標準の1コマの長さ）
    const tracks = new Map();
    for (const tr of mk.filter((b) => b.t === 'trak')) {
      const tk = children(v, tr.o + 8, tr.o + tr.size);
      const tkhd = find(tk, 'tkhd'), mdia = find(tk, 'mdia');
      if (!tkhd || !mdia) continue;
      const mdhd = find(children(v, mdia.o + 8, mdia.o + mdia.size), 'mdhd');
      if (!mdhd) continue;
      const tkV = v.getUint8(tkhd.o + 8), mdV = v.getUint8(mdhd.o + 8);
      const id = u32(v, tkhd.o + 8 + (tkV === 1 ? 20 : 12));
      const scale = u32(v, mdhd.o + 8 + (mdV === 1 ? 20 : 12));
      tracks.set(id, { tkhd, tkV, mdhd, mdV, scale, defDur: 0, end: 0, start: Infinity });
    }
    for (const tx of children(v, mvex.o + 8, mvex.o + mvex.size).filter((b) => b.t === 'trex')) {
      const t = tracks.get(u32(v, tx.o + 12));
      if (t) t.defDur = u32(v, tx.o + 20);
    }
    if (children(v, mvex.o + 8, mvex.o + mvex.size).some((b) => b.t === 'mehd')) return blob;
    // かたまりごとに、始まりの時刻と長さを足して、トラックの終わりを求める
    let absolute = false;
    for (const mf of top.filter((b) => b.t === 'moof')) {
      for (const tf of children(v, mf.o + 8, mf.o + mf.size).filter((b) => b.t === 'traf')) {
        const kids = children(v, tf.o + 8, tf.o + tf.size);
        const tfhd = find(kids, 'tfhd');
        if (!tfhd) continue;
        const fl = u32(v, tfhd.o + 8) & 0xffffff;
        const t = tracks.get(u32(v, tfhd.o + 12));
        if (!t) continue;
        if (fl & 0x01) absolute = true;   // ファイルの先頭からの位置で書かれている（ずらせない）
        let p = tfhd.o + 16;
        if (fl & 0x01) p += 8;
        if (fl & 0x02) p += 4;
        const def = fl & 0x08 ? u32(v, p) : t.defDur;
        const tfdt = find(kids, 'tfdt');
        let base = 0;
        if (tfdt) base = v.getUint8(tfdt.o + 8) === 1 ? Number(v.getBigUint64(tfdt.o + 12)) : u32(v, tfdt.o + 12);
        let sum = 0;
        for (const tr of kids.filter((b) => b.t === 'trun')) {
          const tfl = u32(v, tr.o + 8) & 0xffffff, n = u32(v, tr.o + 12);
          let q = tr.o + 16;
          if (tfl & 0x01) q += 4;
          if (tfl & 0x04) q += 4;
          const per = ((tfl & 0x100) ? 4 : 0) + ((tfl & 0x200) ? 4 : 0) + ((tfl & 0x400) ? 4 : 0) + ((tfl & 0x800) ? 4 : 0);
          for (let i = 0; i < n; i++) sum += (tfl & 0x100) ? u32(v, q + i * per) : def;
        }
        t.end = Math.max(t.end, base + sum);
        t.start = Math.min(t.start, base);
      }
    }
    // トラックの長さ＝最後のかたまりの終わり−最初のかたまりの始まり
    for (const t of tracks.values()) t.len = t.end && Number.isFinite(t.start) ? t.end - t.start : 0;
    let sec = 0;
    for (const t of tracks.values()) if (t.scale && t.len) sec = Math.max(sec, t.len / t.scale);
    // 録画した時間と大きく違えば、計算をあきらめて録画した時間を使う
    if (expectedSec > 0 && (!sec || Math.abs(sec - expectedSec) > Math.max(2, expectedSec * 0.1))) {
      sec = expectedSec;
      for (const t of tracks.values()) t.len = Math.round(expectedSec * t.scale);
    }
    const movieDur = Math.round(sec * movieScale);
    if (!movieDur || movieDur > 0xffffffff) return blob;
    // 頭の欄に長さを書く（元の並びは変えずに、値だけ書きかえる）
    const out = new Uint8Array(buf.byteLength + (absolute ? 0 : 16));
    const src = new Uint8Array(buf);
    const setDur = (dv, o, ver, at0, at1, val) => {
      if (ver === 1) dv.setBigUint64(o + 8 + at1, BigInt(val)); else dv.setUint32(o + 8 + at0, Math.min(val, 0xffffffff));
    };
    const w = new DataView(src.buffer.slice(0));
    setDur(w, mvhd.o, mvV, 16, 24, movieDur);
    for (const t of tracks.values()) {
      if (!t.len) continue;
      setDur(w, t.tkhd.o, t.tkV, 20, 28, Math.round(t.len / t.scale * movieScale));
      setDur(w, t.mdhd.o, t.mdV, 16, 24, t.len);
    }
    const fixed = new Uint8Array(w.buffer);
    if (absolute) return new Blob([fixed], { type: blob.type });
    // かたまりつなぎ全体の長さ（mehd）を mvex の頭に加える。後ろはそのままずれるだけ（位置はかたまりの頭からの相対）
    const ins = mvex.o + 8;
    out.set(fixed.subarray(0, ins), 0);
    const m = new DataView(out.buffer);
    m.setUint32(ins, 16); out.set([0x6d, 0x65, 0x68, 0x64], ins + 4); m.setUint32(ins + 8, 0); m.setUint32(ins + 12, movieDur);
    out.set(fixed.subarray(ins), ins + 16);
    m.setUint32(moov.o, moov.size + 16);
    m.setUint32(mvex.o, mvex.size + 16);
    return new Blob([out], { type: blob.type });
  } catch (e) {
    return blob;   // 直せないときは元のまま
  }
}
