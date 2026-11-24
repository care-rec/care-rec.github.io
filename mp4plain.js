// かたまりつなぎの mp4 を、ふつうの1本の mp4 に組み直す。
// iPhone の録画（MediaRecorder）は、短いかたまり（fragment）をつないだ mp4 を作る。iPhone の再生部品は、
// この形では頭に書いた長さを使わず、自分で推し量った長さ（1分1秒の動画を2分11秒、1分6秒を28秒など）で
// 再生するため、確認画面・手動ぼかし・保存した動画がおかしくなる。
// ふつうの mp4（目次＝moov に全部のコマの位置・長さが書いてある形）なら、正しい長さで扱われる。
// 映像と音のデータそのものは変えず、並べ方と目次だけを作り直す。
import { fixMp4Duration } from './mp4fix.js';

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

function box(typeStr, ...parts) {
  let len = 8;
  for (const p of parts) len += p.byteLength;
  const out = new Uint8Array(len);
  new DataView(out.buffer).setUint32(0, len);
  for (let i = 0; i < 4; i++) out[4 + i] = typeStr.charCodeAt(i);
  let o = 8;
  for (const p of parts) { out.set(p, o); o += p.byteLength; }
  return out;
}
function u32s(vals) {
  const a = new Uint8Array(vals.length * 4), dv = new DataView(a.buffer);
  for (let i = 0; i < vals.length; i++) dv.setUint32(i * 4, vals[i] >>> 0);
  return a;
}
// 版とフラグ（0）＋本体
const full = (typeStr, vals) => box(typeStr, u32s([0, ...vals]));

// 組み直した mp4 のバイト列。組み直せないときは null
export function plainMp4(buf, expectedSec = 0) {
  const v = new DataView(buf);
  const src = new Uint8Array(buf);
  const top = children(v, 0, buf.byteLength);
  const ftyp = find(top, 'ftyp'), moov = find(top, 'moov');
  if (!ftyp || !moov || !top.some((b) => b.t === 'moof')) return null;
  const mk = children(v, moov.o + 8, moov.o + moov.size);
  const mvhd = find(mk, 'mvhd'), mvex = find(mk, 'mvex');
  if (!mvhd) return null;
  const mvV = v.getUint8(mvhd.o + 8);
  const movieScale = u32(v, mvhd.o + 8 + (mvV === 1 ? 20 : 12));
  if (!movieScale) return null;
  // トラックごと
  const tracks = new Map();
  for (const tr of mk.filter((b) => b.t === 'trak')) {
    const tk = children(v, tr.o + 8, tr.o + tr.size);
    const tkhd = find(tk, 'tkhd'), mdia = find(tk, 'mdia');
    if (!tkhd || !mdia) return null;
    const md = children(v, mdia.o + 8, mdia.o + mdia.size);
    const mdhd = find(md, 'mdhd'), hdlr = find(md, 'hdlr'), minf = find(md, 'minf');
    if (!mdhd || !hdlr || !minf) return null;
    const mi = children(v, minf.o + 8, minf.o + minf.size);
    const stbl = find(mi, 'stbl');
    if (!stbl) return null;
    const stsd = find(children(v, stbl.o + 8, stbl.o + stbl.size), 'stsd');
    if (!stsd) return null;
    const tkV = v.getUint8(tkhd.o + 8), mdV = v.getUint8(mdhd.o + 8);
    const id = u32(v, tkhd.o + 8 + (tkV === 1 ? 20 : 12));
    tracks.set(id, {
      tk, tkhd, tkV, md, mdhd, mdV, mi, stsd,
      handler: type(v, hdlr.o + 12),   // 'vide'（映像）か 'soun'（音）
      scale: u32(v, mdhd.o + 8 + (mdV === 1 ? 20 : 12)),
      defDur: 0, defSize: 0, defFlags: 0, samples: [], chunks: [],
    });
  }
  if (mvex) {
    for (const tx of children(v, mvex.o + 8, mvex.o + mvex.size).filter((b) => b.t === 'trex')) {
      const t = tracks.get(u32(v, tx.o + 12));
      if (t) { t.defDur = u32(v, tx.o + 20); t.defSize = u32(v, tx.o + 24); t.defFlags = u32(v, tx.o + 28); }
    }
  }
  // かたまりを順に読んで、コマ（sample）の位置・大きさ・長さ・区切りのコマ（キーフレーム）かを集める
  for (const mf of top.filter((b) => b.t === 'moof')) {
    for (const tf of children(v, mf.o + 8, mf.o + mf.size).filter((b) => b.t === 'traf')) {
      const kids = children(v, tf.o + 8, tf.o + tf.size);
      const tfhd = find(kids, 'tfhd');
      if (!tfhd) return null;
      const fl = u32(v, tfhd.o + 8) & 0xffffff;
      const t = tracks.get(u32(v, tfhd.o + 12));
      if (!t) return null;
      let p = tfhd.o + 16, baseOff = mf.o;
      if (fl & 0x01) { baseOff = Number(v.getBigUint64(p)); p += 8; }
      if (fl & 0x02) p += 4;
      let dDur = t.defDur, dSize = t.defSize, dFlags = t.defFlags;
      if (fl & 0x08) { dDur = u32(v, p); p += 4; }
      if (fl & 0x10) { dSize = u32(v, p); p += 4; }
      if (fl & 0x20) { dFlags = u32(v, p); p += 4; }
      for (const tr of kids.filter((b) => b.t === 'trun')) {
        const tfl = u32(v, tr.o + 8) & 0xffffff, n = u32(v, tr.o + 12);
        if (tfl & 0x800) return null;   // 表示の順と並びの順が違う映像（Bフレーム）は扱わない
        let q = tr.o + 16, dataOff = baseOff, first = null;
        if (tfl & 0x01) { dataOff = baseOff + v.getInt32(q); q += 4; }
        if (tfl & 0x04) { first = u32(v, q); q += 4; }
        const chunk = { off: dataOff, size: 0, n };
        let o = dataOff;
        for (let i = 0; i < n; i++) {
          let dur = dDur, size = dSize, flags = dFlags;
          if (tfl & 0x100) { dur = u32(v, q); q += 4; }
          if (tfl & 0x200) { size = u32(v, q); q += 4; }
          if (tfl & 0x400) { flags = u32(v, q); q += 4; }
          if (i === 0 && first !== null) flags = first;
          t.samples.push({ size, dur, sync: !(flags & 0x10000) });
          o += size;
          chunk.size += size;
        }
        if (o > buf.byteLength) return null;
        if (n) t.chunks.push(chunk);
      }
    }
  }
  const list = [...tracks.values()];
  if (!list.length || list.some((t) => !t.samples.length || !t.scale)) return null;
  for (const t of list) t.total = t.samples.reduce((a, s) => a + s.dur, 0);
  // 音が映像より長く続いているとき（書き出しの止まり際に音だけ録れた分）は、映像の終わりで音を切る。
  // 切らないと、最後の場面が止まったまま数秒続く
  const vid = list.find((t) => t.handler === 'vide');
  if (vid) {
    const vEnd = vid.total / vid.scale;
    for (const t of list) {
      if (t === vid || t.total / t.scale <= vEnd + 0.1) continue;
      let acc = 0, keep = 0;
      while (keep < t.samples.length && (acc + t.samples[keep].dur) / t.scale <= vEnd + 0.05) acc += t.samples[keep++].dur;
      if (keep < 1) continue;
      t.samples.length = keep;
      t.total = acc;
      // チャンクも、残したコマの分だけにする
      let left = keep;
      const kept = [];
      let si = 0;
      for (const c of t.chunks) {
        if (left <= 0) break;
        if (c.n <= left) { kept.push(c); left -= c.n; si += c.n; continue; }
        let sz = 0;
        for (let i = 0; i < left; i++) sz += t.samples[si + i].size;
        kept.push({ off: c.off, size: sz, n: left });
        left = 0;
      }
      t.chunks = kept;
    }
  }
  const movieDur = Math.max(...list.map((t) => Math.round(t.total / t.scale * movieScale)));
  // 録画した時間とかけ離れているときは、読み方がおかしいので組み直さない（コマごとの長さを足しているので、多少の差は正しい）
  if (expectedSec > 0 && Math.abs(movieDur / movieScale - expectedSec) > Math.max(5, expectedSec * 0.3)) return null;
  // 新しいデータの並び：元のファイルでの順（音と映像が交互に並んだまま）
  const chunks = list.flatMap((t) => t.chunks).sort((a, b) => a.off - b.off);
  let mdatLen = 0;
  for (const c of chunks) { c.rel = mdatLen; mdatLen += c.size; }
  if (mdatLen + 1e7 > 0xffffffff) return null;
  // 目次（moov）を作る。チャンクの位置は、moov の大きさが決まってから入れる
  const sub = (b) => src.subarray(b.o, b.o + b.size);
  const buildMoov = (mdatStart) => {
    const traks = list.map((t) => {
      const stts = [];   // 同じ長さのコマが続く所をまとめる
      for (const s of t.samples) {
        const last = stts[stts.length - 1];
        if (last && last[1] === s.dur) last[0]++; else stts.push([1, s.dur]);
      }
      const stsc = [];   // チャンクごとのコマ数が変わる所だけ書く
      t.chunks.forEach((c, i) => { const last = stsc[stsc.length - 1]; if (!last || last[1] !== c.n) stsc.push([i + 1, c.n, 1]); });
      const parts = [sub(t.stsd), full('stts', [stts.length, ...stts.flat()])];
      if (t.handler === 'vide' && t.samples.some((s) => !s.sync)) {
        const sync = [];
        t.samples.forEach((s, i) => { if (s.sync) sync.push(i + 1); });
        parts.push(full('stss', [sync.length, ...sync]));
      }
      parts.push(
        full('stsc', [stsc.length, ...stsc.flat()]),
        full('stsz', [0, t.samples.length, ...t.samples.map((s) => s.size)]),
        full('stco', [t.chunks.length, ...t.chunks.map((c) => mdatStart + c.rel)]),
      );
      const stbl = box('stbl', ...parts);
      const minf = box('minf', ...t.mi.filter((b) => b.t !== 'stbl').map(sub), stbl);
      const mdhd = src.slice(t.mdhd.o, t.mdhd.o + t.mdhd.size);
      const tkhd = src.slice(t.tkhd.o, t.tkhd.o + t.tkhd.size);
      const dm = new DataView(mdhd.buffer), dt = new DataView(tkhd.buffer);
      if (t.mdV === 1) dm.setBigUint64(8 + 24, BigInt(t.total)); else dm.setUint32(8 + 16, t.total);
      const td = Math.round(t.total / t.scale * movieScale);
      if (t.tkV === 1) dt.setBigUint64(8 + 28, BigInt(td)); else dt.setUint32(8 + 20, td);
      const mdia = box('mdia', mdhd, ...t.md.filter((b) => b.t !== 'mdhd' && b.t !== 'minf').map(sub), minf);
      return box('trak', tkhd, ...t.tk.filter((b) => b.t !== 'tkhd' && b.t !== 'mdia').map(sub), mdia);
    });
    const mv = src.slice(mvhd.o, mvhd.o + mvhd.size);
    const dmv = new DataView(mv.buffer);
    if (mvV === 1) dmv.setBigUint64(8 + 24, BigInt(movieDur)); else dmv.setUint32(8 + 16, movieDur);
    const rest = mk.filter((b) => b.t !== 'mvhd' && b.t !== 'trak' && b.t !== 'mvex').map(sub);
    return box('moov', mv, ...traks, ...rest);
  };
  const ftypBytes = sub(ftyp);
  const size0 = buildMoov(0).byteLength;
  const mdatStart = ftypBytes.byteLength + size0 + 8;
  const moovBytes = buildMoov(mdatStart);
  if (moovBytes.byteLength !== size0) return null;
  const out = new Uint8Array(mdatStart + mdatLen);
  out.set(ftypBytes, 0);
  out.set(moovBytes, ftypBytes.byteLength);
  new DataView(out.buffer).setUint32(mdatStart - 8, mdatLen + 8);
  out.set([0x6d, 0x64, 0x61, 0x74], mdatStart - 4);   // 'mdat'
  for (const c of chunks) out.set(src.subarray(c.off, c.off + c.size), mdatStart + c.rel);
  return out;
}

// 録画した mp4 を、iPhone が正しい長さで扱える形にする。
// まずふつうの1本の mp4 に組み直し、できなければ頭に長さを書く方法にする
export async function normalizeMp4(blob, expectedSec = 0) {
  try {
    const plain = plainMp4(await blob.arrayBuffer(), expectedSec);
    if (plain) return new Blob([plain], { type: blob.type });
  } catch (e) { /* 長さだけ直す方法へ */ }
  return fixMp4Duration(blob, expectedSec);
}
