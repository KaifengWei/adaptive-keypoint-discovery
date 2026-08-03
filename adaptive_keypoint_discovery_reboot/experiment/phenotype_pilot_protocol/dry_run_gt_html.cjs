const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");

async function clickImageFraction(page, fx, fy) {
  const box = await page.locator("#overlay").boundingBox();
  if (!box) throw new Error("overlay has no bounding box");
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
}

async function traceOne(page, variant) {
  const initial = await page.evaluate(() => window.__GT_DEBUG__.getState());
  if (initial.record.traces.length !== 0) throw new Error("practice image did not start with zero traces");
  await page.click("#setBaseBtn");
  await clickImageFraction(page, variant ? 0.78 : 0.82, variant ? 0.55 : 0.48);
  await page.click("#addTraceBtn");
  await page.selectOption("#visibilityStatus", "measurable");
  await page.selectOption("#confidence", "high");
  await clickImageFraction(page, 0.66, variant ? 0.52 : 0.47);
  await clickImageFraction(page, 0.48, variant ? 0.47 : 0.45);
  await clickImageFraction(page, 0.24, variant ? 0.38 : 0.43);
  await page.click("#submitPlantBtn");
  const submitted = await page.evaluate(() => window.__GT_DEBUG__.getState());
  const trace = submitted.record.traces[0];
  if (!submitted.record.submitted || trace.gt_leaf_id_postsubmit !== "GT01" || !trace.reference_main_path || !(trace.structural_path_length_px > 0)) {
    throw new Error("submit/GT/main-path/length dry run failed");
  }
  return { initial_trace_count: initial.record.traces.length, submitted_trace_count: 1, gt_assigned_after_submit: true, structural_path_length_positive: true };
}

(async () => {
  const protocolDir = __dirname;
  const entry = path.join(protocolDir, "runtime", "packages", "practice", "index.html");
  const edge = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const reportDir = path.join(protocolDir, "validation", "20260803");
  fs.mkdirSync(reportDir, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: edge });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text()); });
  await page.goto(pathToFileURL(entry).href);
  await page.waitForFunction(() => window.__GT_DEBUG__ && document.querySelector("#plantImage").complete);
  const first = await traceOne(page, 0);
  await page.click("#nextBtn");
  await page.waitForFunction(() => document.querySelector("#plantImage").complete);
  const second = await traceOne(page, 1);
  const bundle = await page.evaluate(async () => {
    const result = await window.__GT_DEBUG__.buildExportBundle(true);
    return { sessionRows: result.sessionRows, traceRows: result.traceRows, payloadHash: result.payload.export_sha256 };
  });
  if (bundle.sessionRows.length !== 2 || bundle.traceRows.length !== 2 || !/^[0-9a-f]{64}$/.test(bundle.payloadHash)) throw new Error("anonymous export bundle failed");
  const serialized = JSON.stringify(bundle).toLowerCase();
  const forbidden = ["v4_val_", "dataset_id", "pilot_group", "visible_leaf_count", "leaf_exists", "method_id", "teacher_direct", "student_b", "student_d", "decoder"];
  const hits = forbidden.filter((token) => serialized.includes(token));
  if (hits.length) throw new Error(`dry-run export leakage: ${hits.join(",")}`);
  await page.reload();
  await page.waitForFunction(() => window.__GT_DEBUG__);
  const restored = await page.evaluate(() => window.__GT_DEBUG__.getState());
  if (!restored.record.submitted) throw new Error("localStorage restore failed after reload");
  await page.screenshot({ path: path.join(reportDir, "practice_dry_run_screenshot.png"), fullPage: true });
  const formalPage = await context.newPage();
  const formalEntry = path.join(protocolDir, "runtime", "packages", "rater1_round1", "index.html");
  await formalPage.goto(pathToFileURL(formalEntry).href);
  await formalPage.waitForFunction(() => window.__GT_DEBUG__ && document.querySelector("#plantImage").complete);
  const formalInitial = await formalPage.evaluate(() => window.__GT_DEBUG__.getState());
  if (formalInitial.pack.item_count !== 16 || formalInitial.record.traces.length !== 0 || formalInitial.record.submitted) {
    throw new Error("formal Rater 1 round 1 entry is not a clean 16-image package");
  }
  const browserGeometryFixture = await formalPage.evaluate(() => {
    const value = window.__GT_DEBUG__.geometry.metrics([{x:0,y:0},{x:40,y:20},{x:80,y:10},{x:120,y:50}], 200);
    return {
      structural_path_length_px: value.structural_path_length_px,
      structural_path_length_bbox_norm: value.structural_path_length_bbox_norm,
      chord_length_px: value.chord_length_px,
      total_turning_angle_deg: value.total_turning_angle_deg,
      mean_abs_curvature_per_px: value.mean_abs_curvature_per_px,
    };
  });
  await browser.close();
  if (errors.length) throw new Error(`browser errors: ${errors.join(" | ")}`);
  const report = {
    status: "pass",
    checked_at_utc: new Date().toISOString(),
    browser: "Microsoft Edge headless via Playwright",
    non_pilot_images: 2,
    image_1: first,
    image_2: second,
    export_session_rows: bundle.sessionRows.length,
    export_trace_rows: bundle.traceRows.length,
    export_sha256_present: true,
    export_forbidden_hits: 0,
    local_storage_restore: true,
    formal_rater1_round1_initial_state: "16 images, zero traces, unsubmitted",
    browser_geometry_fixture: browserGeometryFixture,
    browser_errors: 0,
    test_images_read: 0,
    model_outputs_read: 0,
  };
  fs.writeFileSync(path.join(reportDir, "non_pilot_dry_run_report.json"), JSON.stringify(report, null, 2), "utf8");
  process.stdout.write(JSON.stringify(report));
})().catch((error) => { process.stderr.write(String(error.stack || error)); process.exit(1); });
