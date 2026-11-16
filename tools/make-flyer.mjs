// 配布用の案内（A4）を、PDFと画像で書き出す。
//   node tools/make-flyer.mjs [URL]
// URLを変えたときは、QRコードを作り直してからこれを実行する。
// 必要なもの：puppeteer-core（作業用フォルダに入れたものを NODE_PATH で指定）と Chrome。
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = path.join(root, '配布用');
const url = process.argv[2] || 'https://care-rec.github.io/';
const src = fs.readFileSync(path.join(dir, '案内チラシ.html'), 'utf8').replaceAll('{{URL}}', url);
const tmp = path.join(dir, '_build.html');
fs.writeFileSync(tmp, src);

const browser = await puppeteer.launch({ executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', headless: 'new' });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 794, height: 1123, deviceScaleFactor: 2 });
  await page.goto(pathToFileURL(tmp).href, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  const over = await page.evaluate(() => { const p = document.querySelector('.page'); return p.scrollHeight - p.clientHeight; });
  console.log('はみ出し(px):', over, over > 0 ? '← 1枚に収まっていません' : '（1枚に収まっています）');
  await page.pdf({ path: path.join(dir, 'ケアREC_案内.pdf'), format: 'A4', printBackground: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });
  await page.screenshot({ path: path.join(dir, 'ケアREC_案内.png'), clip: { x: 0, y: 0, width: 794, height: 1123 } });
  console.log('書き出し完了:', path.join(dir, 'ケアREC_案内.pdf'));
} finally {
  await browser.close();
  fs.rmSync(tmp, { force: true });
}
