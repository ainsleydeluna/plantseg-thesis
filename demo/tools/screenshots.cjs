// Screenshots of what the panel sees, from headless Chromium emulating an iPhone 15 Plus viewport
// (430 x 739 CSS px, DPR 3). Chromium is not Safari: layout only, not a Safari compatibility test.
//
//   node demo/tools/screenshots.cjs http://127.0.0.1:8000 IMAGE OUT_DIR
"use strict";
const path = require("path");
const { chromium, devices } = require("playwright");

async function main() {
  const [base, image, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ ...devices["iPhone 15 Plus"] });
  const page = await ctx.newPage();
  const shot = (name, full) => page.screenshot({ path: path.join(out, name), type: "jpeg", quality: 80, fullPage: full });

  await page.goto(base + "/");
  await page.waitForFunction(() => document.getElementById("modelSha").textContent.length > 10);
  await shot("01_server_page.jpg", false);
  await page.setInputFiles("#inUpload", image);
  await page.waitForFunction(() => !document.getElementById("result").hidden && document.getElementById("latInfer").textContent !== "–");
  await page.waitForTimeout(800);
  await page.evaluate(() => window.scrollTo(0, 0));
  await shot("02_server_result.jpg", true);

  await page.goto(base + "/ondevice/?model=int8&ep=wasm");
  await page.waitForFunction(() => window.__demo && (window.__demo.ready || window.__demo.error), null, { timeout: 180000 });
  await page.setInputFiles("#inUpload", image);
  await page.waitForFunction(() => window.__demo.results.length > 0, null, { timeout: 180000 });
  await page.waitForTimeout(500);
  await page.evaluate(() => window.scrollTo(0, 0));
  await shot("03_ondevice_result.jpg", true);
  await browser.close();
  console.log("written to " + out);
}
main().catch((e) => { console.error(e); process.exit(1); });
