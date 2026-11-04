// Headless check of the in-browser (ONNX Runtime Web) page against the evaluator, in Chromium.
//
//   node demo/tools/browser_check.cjs http://127.0.0.1:8000 FIXTURE_DIR [wasm|webgpu ...]
//
// FIXTURE_DIR comes from `python3 -I demo/tools/make_browser_fixtures.py FIXTURE_DIR`. For each model
// (INT8 QDQ, FP32) and execution provider it loads /ondevice/, feeds every fixture through the page's file
// input and compares: the page's input tensor sha256 with the evaluator's; the page's cropped mask sha256
// with the PyTorch model's; and, for the first fixture, the page's logits with the PyTorch logits.
// Chromium emulates the iPhone 15 Plus viewport only: it is not Safari. Prints one JSON document.
"use strict";
const fs = require("fs");
const path = require("path");
const { chromium, devices } = require("playwright");

async function main() {
  const [base, dir, ...eps] = process.argv.slice(2);
  const expected = JSON.parse(fs.readFileSync(path.join(dir, "expected.json"), "utf8"));
  const report = { base, chromium: null, runs: [] };
  const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan",
    "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"] });
  report.chromium = browser.version();
  for (const ep of (eps.length ? eps : ["wasm"])) {
    for (const model of ["int8", "fp32"]) {
      const ctx = await browser.newContext({ ...devices["iPhone 15 Plus"] });
      const page = await ctx.newPage();
      const logs = [];
      page.on("console", (m) => { if (logs.length < 40) logs.push(m.type() + ": " + m.text().slice(0, 300)); });
      page.on("pageerror", (e) => logs.push("pageerror: " + String(e).slice(0, 300)));
      await page.route("**/__ref/*", (route) => route.fulfill({ path: path.join(dir, path.basename(route.request().url())),
        contentType: "application/octet-stream" }));
      const run = { ep, model, cases: [], logits: null, error: null };
      try {
        await page.goto(`${base}/ondevice/?model=${model}&ep=${ep}`);
        await page.waitForFunction(() => window.__demo && (window.__demo.ready || window.__demo.error), null, { timeout: 180000 });
        run.error = await page.evaluate(() => window.__demo.error || null);
        run.runtime = await page.evaluate(() => document.getElementById("mRuntime").textContent);
        run.model_sha256_in_browser = await page.evaluate(() => window.__demo.modelSha256);
        if (!run.error) {
          for (const c of expected.cases) {
            const n = await page.evaluate(() => window.__demo.results.length);
            await page.setInputFiles("#inUpload", path.join(dir, c.file));
            await page.waitForFunction((k) => window.__demo.results.length > k || document.getElementById("error").style.display === "block", n, { timeout: 180000 });
            const r = await page.evaluate(() => window.__demo.results[window.__demo.results.length - 1] || null);
            const err = await page.evaluate(() => document.getElementById("error").style.display === "block" ? document.getElementById("error").textContent : null);
            run.cases.push({ file: c.file, error: err,
              tensor_equal: !!r && r.tensor_sha256 === c.tensor_sha256,
              box_equal: !!r && JSON.stringify(r.box) === JSON.stringify(c.box),
              mask_equal: !!r && r.mask_crop_sha256 === c[`mask_crop_sha256_${model}`],
              inference_ms: r && Math.round(r.inference_ms), decoder: r && r.decoder, run: r && r.run });
            if (c.file === expected.ref_logits_for) {
              run.logits = await page.evaluate((a) => window.__demo.compareLogits(a[0], a[1]),
                [`/__ref/ref_logits_${model}.f32`, expected.ref_logits_stride]);
            }
          }
          if (ep === "wasm" && model === "int8") {
            await page.screenshot({ path: path.join(dir, `ondevice_${model}_${ep}.png`), fullPage: true });
          }
        }
      } catch (e) { run.error = String(e).slice(0, 500); }
      run.console = logs.slice(0, 12);
      report.runs.push(run);
      await ctx.close();
    }
  }
  await browser.close();
  console.log(JSON.stringify(report, null, 2));
}
main().catch((e) => { console.error(e); process.exit(1); });
