(() => {
  "use strict";

  const PROTOCOL = "phenotype-first-pilot-v2";
  const GEOMETRY = "phenotype-geometry-v1";
  const RESAMPLING = "pchip_arc_length_240";
  const ANGLE_PROTOCOL = "shared_prefix_tau_0.01D_local_interval_0.05D";
  const COLORS = ["#0b84f3", "#e55353", "#16a36a", "#8a5cf6", "#e28b18", "#00a6a6", "#c33a94", "#6d7b8a"];
  const pack = window.PACKAGE_DATA;
  if (!pack || !Array.isArray(pack.items) || !pack.items.length) throw new Error("Invalid anonymous package data");

  const el = (id) => document.getElementById(id);
  const ui = {
    packageLabel: el("packageLabel"), positionLabel: el("positionLabel"), blindLabel: el("blindLabel"),
    submissionBadge: el("submissionBadge"), image: el("plantImage"), overlay: el("overlay"), stage: el("stage"),
    prev: el("prevBtn"), next: el("nextBtn"), exportProgress: el("exportProgressBtn"),
    importInput: el("importInput"), exportFinal: el("exportFinalBtn"), setBase: el("setBaseBtn"),
    addTrace: el("addTraceBtn"), drawMode: el("drawModeBtn"), undo: el("undoBtn"), reset: el("resetPlantBtn"),
    modeLabel: el("modeLabel"), baseSummary: el("baseSummary"), traceCount: el("traceCountSummary"),
    measurableCount: el("measurableCountSummary"), traceList: el("traceList"), traceEditor: el("traceEditor"),
    visibility: el("visibilityStatus"), occlusion: el("occlusionGrade"), interpolation: el("interpolationUsed"),
    confidence: el("confidence"), traceNotes: el("traceNotes"), metricPreview: el("metricPreview"),
    deleteTrace: el("deleteTraceBtn"), plantNotes: el("plantNotes"), submit: el("submitPlantBtn"),
    revise: el("revisePlantBtn"), validation: el("validationMessage"),
    downloadArea: el("downloadArea"),
  };

  let current = 0;
  let mode = "browse";
  let activeTrace = null;
  let drag = null;
  let downloadUrls = [];
  const storageKey = `phenotype-gt-v2-${pack.package_id}`;
  let records = loadRecords();

  function uuid() {
    if (crypto.randomUUID) return crypto.randomUUID();
    const bytes = new Uint8Array(16); crypto.getRandomValues(bytes);
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  function defaultRecord(item) {
    return {
      protocol_version: PROTOCOL, geometry_protocol_version: GEOMETRY, package_id: pack.package_id,
      blind_id: item.blind_id, rater_id: pack.rater_id, measurement_round: pack.measurement_round,
      randomized_order: item.randomized_order, image_alias: item.image_alias, plant_submission_uuid: uuid(),
      base: null, traces: [], plant_notes: "", submitted: false, submission_revision: 1,
      submitted_at_utc: "", gt_freeze_status: "not_frozen",
    };
  }

  function loadRecords() {
    try {
      const parsed = JSON.parse(localStorage.getItem(storageKey) || "{}");
      const out = {};
      for (const item of pack.items) out[item.blind_id] = parsed[item.blind_id] || defaultRecord(item);
      return out;
    } catch (_) {
      const out = {}; for (const item of pack.items) out[item.blind_id] = defaultRecord(item); return out;
    }
  }

  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(records)); } catch (_) {}
  }

  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function cleanPoints(points) {
    const result = [];
    for (const p of points) if (!result.length || distance(p, result[result.length - 1]) > 1e-6) result.push({ x: +p.x, y: +p.y });
    if (result.length < 2) throw new Error("轨迹至少需要两个不同点");
    return result;
  }

  function edgeSlope(h0, h1, m0, m1) {
    let value = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (value === 0 || m0 === 0 || Math.sign(value) !== Math.sign(m0)) return 0;
    if (Math.sign(m0) !== Math.sign(m1) && Math.abs(value) > 3 * Math.abs(m0)) value = 3 * m0;
    return value;
  }

  function pchipDerivatives(x, y) {
    const n = x.length, h = [], slopes = [];
    for (let i = 0; i < n - 1; i++) { h.push(x[i + 1] - x[i]); slopes.push((y[i + 1] - y[i]) / h[i]); }
    if (n === 2) return [slopes[0], slopes[0]];
    const d = new Array(n).fill(0);
    d[0] = edgeSlope(h[0], h[1], slopes[0], slopes[1]);
    d[n - 1] = edgeSlope(h[n - 2], h[n - 3], slopes[n - 2], slopes[n - 3]);
    for (let k = 1; k < n - 1; k++) {
      const left = slopes[k - 1], right = slopes[k];
      if (left === 0 || right === 0 || Math.sign(left) !== Math.sign(right)) d[k] = 0;
      else {
        const w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1];
        d[k] = (w1 + w2) / (w1 / left + w2 / right);
      }
    }
    return d;
  }

  function pchipEval(x, y, target) {
    const d = pchipDerivatives(x, y), out = [];
    let k = 0;
    for (const value of target) {
      while (k < x.length - 2 && value > x[k + 1]) k++;
      const h = x[k + 1] - x[k], t = (value - x[k]) / h;
      const h00 = 2 * t ** 3 - 3 * t ** 2 + 1;
      const h10 = t ** 3 - 2 * t ** 2 + t;
      const h01 = -2 * t ** 3 + 3 * t ** 2;
      const h11 = t ** 3 - t ** 2;
      out.push(h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]);
    }
    return out;
  }

  function resampleTrace(raw, count = 240) {
    const points = cleanPoints(raw), cumulative = [0];
    for (let i = 1; i < points.length; i++) cumulative.push(cumulative[i - 1] + distance(points[i], points[i - 1]));
    const total = cumulative[cumulative.length - 1];
    if (!(total > 1e-12)) throw new Error("轨迹长度为0");
    const target = Array.from({ length: count }, (_, i) => total * i / (count - 1));
    const xs = pchipEval(cumulative, points.map((p) => p.x), target);
    const ys = pchipEval(cumulative, points.map((p) => p.y), target);
    return xs.map((x, i) => ({ x, y: ys[i] }));
  }

  function polylineLength(points) {
    let value = 0; for (let i = 1; i < points.length; i++) value += distance(points[i - 1], points[i]); return value;
  }

  function clockwiseAngle(base, tip) {
    let angle = Math.atan2(tip.y - base.y, tip.x - base.x) * 180 / Math.PI;
    return (angle + 360) % 360;
  }

  function metrics(points, bboxDiag) {
    const curve = resampleTrace(points, 240), length = polylineLength(curve), chord = distance(curve[0], curve[curve.length - 1]);
    let totalTurn = 0;
    for (let i = 1; i < curve.length - 1; i++) {
      const a = { x: curve[i].x - curve[i - 1].x, y: curve[i].y - curve[i - 1].y };
      const b = { x: curve[i + 1].x - curve[i].x, y: curve[i + 1].y - curve[i].y };
      const na = Math.hypot(a.x, a.y), nb = Math.hypot(b.x, b.y);
      if (na > 1e-12 && nb > 1e-12) totalTurn += Math.acos(clamp((a.x * b.x + a.y * b.y) / (na * nb), -1, 1));
    }
    return {
      curve, structural_path_length_px: length, structural_path_length_bbox_norm: length / Math.max(bboxDiag, 1e-12),
      chord_length_px: chord, tip_clockwise_angle_deg: clockwiseAngle(curve[0], curve[curve.length - 1]),
      total_turning_angle_deg: totalTurn * 180 / Math.PI, mean_abs_curvature_per_px: totalTurn / Math.max(length, 1e-12),
    };
  }

  function cumulative(curve) {
    const result = [0]; for (let i = 1; i < curve.length; i++) result.push(result[i - 1] + distance(curve[i - 1], curve[i])); return result;
  }

  function pointAtArc(curve, cum, target) {
    target = clamp(target, 0, cum[cum.length - 1]);
    let i = 0; while (i < cum.length - 2 && target > cum[i + 1]) i++;
    const span = cum[i + 1] - cum[i], ratio = span <= 1e-12 ? 0 : (target - cum[i]) / span;
    return { x: curve[i].x * (1 - ratio) + curve[i + 1].x * ratio, y: curve[i].y * (1 - ratio) + curve[i + 1].y * ratio };
  }

  function divergence(main, branch, bboxDiag) {
    const tolerance = Math.max(2, 0.01 * bboxDiag), nearestIndex = [], nearestDistance = [];
    for (const p of branch) {
      let best = Infinity, bestIndex = 0;
      for (let j = 0; j < main.length; j++) { const d = distance(p, main[j]); if (d < best) { best = d; bestIndex = j; } }
      nearestIndex.push(bestIndex); nearestDistance.push(best);
    }
    let outside = 0, sharedEnd = null;
    for (let i = 0; i < branch.length; i++) {
      if (nearestDistance[i] <= tolerance) { sharedEnd = i; outside = 0; }
      else if (++outside >= 3) break;
    }
    if (sharedEnd === null || sharedEnd >= branch.length - 3) return { status: "divergence_unresolved", divergence_angle_deg: "" };
    const mainIndex = nearestIndex[sharedEnd], mainCum = cumulative(main), branchCum = cumulative(branch), window = 0.05 * bboxDiag;
    const mainStartS = Math.max(0, mainCum[mainIndex] - window), branchEndS = Math.min(branchCum[branchCum.length - 1], branchCum[sharedEnd] + window);
    const mainStart = pointAtArc(main, mainCum, mainStartS), branchEnd = pointAtArc(branch, branchCum, branchEndS);
    const mv = { x: main[mainIndex].x - mainStart.x, y: main[mainIndex].y - mainStart.y };
    const bv = { x: branchEnd.x - branch[sharedEnd].x, y: branchEnd.y - branch[sharedEnd].y };
    const denominator = Math.hypot(mv.x, mv.y) * Math.hypot(bv.x, bv.y);
    if (denominator <= 1e-12) return { status: "divergence_unresolved", divergence_angle_deg: "" };
    const angle = Math.acos(clamp((mv.x * bv.x + mv.y * bv.y) / denominator, -1, 1)) * 180 / Math.PI;
    return {
      status: "ok", angle_protocol: ANGLE_PROTOCOL, divergence_angle_deg: angle,
      divergence_point_x_px: branch[sharedEnd].x, divergence_point_y_px: branch[sharedEnd].y,
      main_local_interval_px: mainCum[mainIndex] - mainStartS, branch_local_interval_px: branchEndS - branchCum[sharedEnd],
    };
  }

  function chooseMain(computed, bboxDiag) {
    const eligible = computed.filter((x) => x.visibility_status === "measurable");
    if (!eligible.length) return null;
    const maxLength = Math.max(...eligible.map((x) => x.structural_path_length_px));
    let candidates = eligible.filter((x) => (maxLength - x.structural_path_length_px) / Math.max(maxLength, 1e-12) <= 0.01);
    const maxChord = Math.max(...candidates.map((x) => x.chord_length_px)), chordTol = Math.max(1, 0.001 * bboxDiag);
    candidates = candidates.filter((x) => maxChord - x.chord_length_px <= chordTol);
    candidates.sort((a, b) => a.tip_clockwise_angle_deg - b.tip_clockwise_angle_deg || a.trace_uuid.localeCompare(b.trace_uuid));
    return candidates[0].trace_uuid;
  }

  function currentItem() { return pack.items[current]; }
  function currentRecord() { return records[currentItem().blind_id]; }
  function getTrace() { return currentRecord().traces.find((t) => t.trace_uuid === activeTrace) || null; }

  function message(text, kind = "") { ui.validation.textContent = text; ui.validation.className = `validation ${kind}`; }
  function clearDownloads() { downloadUrls.forEach((url) => URL.revokeObjectURL(url)); downloadUrls = []; ui.downloadArea.innerHTML = ""; ui.downloadArea.classList.add("hidden"); }
  function setMode(value) { mode = value; ui.modeLabel.textContent = `当前：${value === "set-base" ? "设置共同基点" : value === "draw" ? "描迹" : "浏览"}`; }

  function imagePoint(event) {
    const rect = ui.overlay.getBoundingClientRect(), item = currentItem();
    return { x: clamp((event.clientX - rect.left) * item.width / rect.width, 0, item.width - 1), y: clamp((event.clientY - rect.top) * item.height / rect.height, 0, item.height - 1) };
  }

  function svgNode(name, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  }

  function renderOverlay() {
    const record = currentRecord(), item = currentItem();
    ui.overlay.setAttribute("viewBox", `0 0 ${item.width} ${item.height}`);
    ui.overlay.innerHTML = "";
    const radius = Math.max(5, item.width * 0.004), lineWidth = Math.max(3, item.width * 0.0025);
    if (record.base) {
      ui.overlay.appendChild(svgNode("circle", { cx: record.base.x, cy: record.base.y, r: radius * 1.35, fill: "#ffd23f", stroke: "#111", "stroke-width": lineWidth * 0.7 }));
    }
    record.traces.forEach((trace, traceIndex) => {
      const color = COLORS[traceIndex % COLORS.length], points = trace.submitted_curve || trace.points;
      if (points.length >= 2) {
        ui.overlay.appendChild(svgNode("polyline", { points: points.map((p) => `${p.x},${p.y}`).join(" "), fill: "none", stroke: color, "stroke-width": lineWidth, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: trace.trace_uuid === activeTrace ? 1 : 0.72 }));
      }
      trace.points.forEach((p, pointIndex) => {
        const circle = svgNode("circle", { cx: p.x, cy: p.y, r: pointIndex === 0 ? radius * 1.05 : radius, fill: pointIndex === trace.points.length - 1 ? "#fff" : color, stroke: color, "stroke-width": lineWidth * 0.65, "data-trace": trace.trace_uuid, "data-point": pointIndex });
        circle.style.cursor = record.submitted || pointIndex === 0 ? "default" : "move";
        ui.overlay.appendChild(circle);
      });
    });
  }

  function renderTraceList() {
    const record = currentRecord(); ui.traceList.innerHTML = "";
    if (!record.traces.length) ui.traceList.innerHTML = '<div class="empty">尚未添加。请依据原图自行判断。</div>';
    record.traces.forEach((trace, index) => {
      const card = document.createElement("div"); card.className = `trace-card ${trace.trace_uuid === activeTrace ? "active" : ""}`; card.style.setProperty("--trace-color", COLORS[index % COLORS.length]);
      const label = trace.gt_leaf_id_postsubmit || `待提交项目 ${index + 1}`;
      card.innerHTML = `<div class="row"><strong>${label}</strong><span>${trace.visibility_status || "未选择可测性"}</span></div><small>控制点 ${trace.points.length}${trace.reference_main_path ? " · 主路径" : ""}</small>`;
      card.addEventListener("click", () => { activeTrace = trace.trace_uuid; render(); }); ui.traceList.appendChild(card);
    });
  }

  function renderEditor() {
    const trace = getTrace(), locked = currentRecord().submitted;
    ui.traceEditor.classList.toggle("hidden", !trace);
    if (!trace) return;
    ui.visibility.value = trace.visibility_status || ""; ui.occlusion.value = trace.occlusion_grade || "none";
    ui.interpolation.value = trace.interpolation_used || "no"; ui.confidence.value = trace.confidence || ""; ui.traceNotes.value = trace.notes || "";
    [ui.visibility, ui.occlusion, ui.interpolation, ui.confidence, ui.traceNotes, ui.deleteTrace].forEach((node) => node.disabled = locked);
    if (trace.points.length >= 2) {
      try {
        const m = metrics(trace.points, currentItem().bbox_diag_px);
        ui.metricPreview.innerHTML = `基部—叶尖结构路径长度：<strong>${m.structural_path_length_px.toFixed(2)} px</strong><br>bbox归一化：${m.structural_path_length_bbox_norm.toFixed(4)}<br>当前仅为页面预览，提交时固定重算。`;
      } catch (_) { ui.metricPreview.textContent = "轨迹尚不足以计算。"; }
    } else ui.metricPreview.textContent = "请继续描迹到叶尖。";
  }

  function render() {
    const item = currentItem(), record = currentRecord();
    ui.packageLabel.textContent = `${pack.display_name} · 协议 ${PROTOCOL}`;
    ui.positionLabel.textContent = `${current + 1} / ${pack.items.length}`; ui.blindLabel.textContent = item.blind_id;
    ui.image.src = item.image_alias; ui.plantNotes.value = record.plant_notes || "";
    ui.baseSummary.textContent = record.base ? `(${record.base.x.toFixed(1)}, ${record.base.y.toFixed(1)})` : "未设置";
    ui.traceCount.textContent = String(record.traces.length);
    ui.measurableCount.textContent = String(record.traces.filter((t) => t.visibility_status === "measurable").length);
    ui.submissionBadge.textContent = record.submitted ? `已提交 r${record.submission_revision}` : `未提交 r${record.submission_revision}`;
    ui.submissionBadge.classList.toggle("done", record.submitted); ui.submit.classList.toggle("hidden", record.submitted); ui.revise.classList.toggle("hidden", !record.submitted);
    [ui.setBase, ui.addTrace, ui.drawMode, ui.undo, ui.reset].forEach((node) => node.disabled = record.submitted);
    ui.prev.disabled = current === 0; ui.next.disabled = current === pack.items.length - 1;
    renderTraceList(); renderEditor(); renderOverlay(); message(""); persist();
  }

  function addTrace() {
    const record = currentRecord(); if (record.submitted) return;
    if (!record.base) return message("请先设置共同地上部基点。", "error");
    const trace = { trace_uuid: uuid(), gt_leaf_id_postsubmit: "", visibility_status: "", trace_status: "incomplete", points: [{ ...record.base }], occlusion_grade: "none", interpolation_used: "no", confidence: "", notes: "", reference_main_path: false };
    clearDownloads(); record.traces.push(trace); activeTrace = trace.trace_uuid; setMode("draw"); render();
  }

  function validateRecord(record) {
    const errors = [];
    if (!record.base) errors.push("未设置共同地上部基点");
    if (!record.traces.length) errors.push("尚未主动添加任何可见叶片");
    record.traces.forEach((trace, index) => {
      const label = `项目${index + 1}`;
      if (!trace.visibility_status) errors.push(`${label}未选择可测性`);
      if (!trace.confidence) errors.push(`${label}未选择信心`);
      if (trace.visibility_status === "measurable" && trace.points.length < 3) errors.push(`${label}可测轨迹至少需要基点、中间点和叶尖`);
      if (trace.visibility_status === "visible_unmeasurable" && trace.points.length < 2) errors.push(`${label}仍需标出叶尖位置`);
      if (trace.visibility_status === "visible_unmeasurable" && !trace.notes.trim()) errors.push(`${label}需要填写不可测原因`);
      if (trace.points.length && distance(trace.points[0], record.base) > 1e-6) errors.push(`${label}第一点不等于共同基点`);
    });
    return errors;
  }

  function submitCurrent() {
    const record = currentRecord(), errors = validateRecord(record);
    if (errors.length) return message(errors.join("\n"), "error");
    const bboxDiag = currentItem().bbox_diag_px, computed = [];
    for (const trace of record.traces) {
      const m = metrics(trace.points, bboxDiag);
      Object.assign(trace, m, { submitted_curve: m.curve, tip_xy: m.curve[m.curve.length - 1] }); delete trace.curve;
      trace.trace_status = trace.visibility_status === "measurable" ? "complete" : "incomplete";
      computed.push(trace);
    }
    const ordered = [...record.traces].sort((a, b) => a.tip_clockwise_angle_deg - b.tip_clockwise_angle_deg || a.trace_uuid.localeCompare(b.trace_uuid));
    ordered.forEach((trace, index) => { trace.gt_leaf_id_postsubmit = `GT${String(index + 1).padStart(2, "0")}`; });
    const mainId = chooseMain(computed, bboxDiag), main = computed.find((trace) => trace.trace_uuid === mainId);
    for (const trace of computed) {
      trace.reference_main_path = trace.trace_uuid === mainId;
      if (!main || trace.trace_uuid === mainId || trace.visibility_status !== "measurable") {
        trace.divergence_status = trace.trace_uuid === mainId ? "not_applicable" : "not_measurable"; trace.divergence_angle_deg = "";
      } else Object.assign(trace, divergence(main.submitted_curve, trace.submitted_curve, bboxDiag));
    }
    record.submitted = true; record.submitted_at_utc = new Date().toISOString(); setMode("browse"); persist(); render(); message("本图已提交并锁定。GT编号和主路径已按预锁定规则生成。", "ok");
  }

  function updateTraceField(field, value) { const trace = getTrace(); if (!trace || currentRecord().submitted) return; clearDownloads(); trace[field] = value; persist(); renderOverlay(); renderTraceList(); }

  ui.setBase.addEventListener("click", () => setMode("set-base")); ui.addTrace.addEventListener("click", addTrace); ui.drawMode.addEventListener("click", () => { if (getTrace()) setMode("draw"); else message("请先添加并选择一个项目。", "error"); });
  ui.undo.addEventListener("click", () => { const trace = getTrace(); if (trace && !currentRecord().submitted && trace.points.length > 1) { clearDownloads(); trace.points.pop(); persist(); render(); } });
  ui.reset.addEventListener("click", () => { if (currentRecord().submitted) return; if (confirm("确定清空本图尚未提交的基点与所有轨迹吗？")) { clearDownloads(); records[currentItem().blind_id] = defaultRecord(currentItem()); activeTrace = null; render(); } });
  ui.deleteTrace.addEventListener("click", () => { const record = currentRecord(); if (record.submitted || !activeTrace) return; clearDownloads(); record.traces = record.traces.filter((t) => t.trace_uuid !== activeTrace); activeTrace = record.traces[0]?.trace_uuid || null; render(); });
  ui.visibility.addEventListener("change", (e) => updateTraceField("visibility_status", e.target.value));
  ui.occlusion.addEventListener("change", (e) => updateTraceField("occlusion_grade", e.target.value));
  ui.interpolation.addEventListener("change", (e) => updateTraceField("interpolation_used", e.target.value));
  ui.confidence.addEventListener("change", (e) => updateTraceField("confidence", e.target.value));
  ui.traceNotes.addEventListener("input", (e) => { const trace = getTrace(); if (trace && !currentRecord().submitted) { clearDownloads(); trace.notes = e.target.value; persist(); } });
  ui.plantNotes.addEventListener("input", (e) => { if (!currentRecord().submitted) { clearDownloads(); currentRecord().plant_notes = e.target.value; persist(); } });
  ui.submit.addEventListener("click", submitCurrent);
  ui.revise.addEventListener("click", () => { const record = currentRecord(); if (!confirm("将保留已提交记录的修订号，并创建可编辑副本。继续吗？")) return; clearDownloads(); record.submitted = false; record.submission_revision += 1; record.submitted_at_utc = ""; record.traces.forEach((t) => { t.gt_leaf_id_postsubmit = ""; t.reference_main_path = false; delete t.submitted_curve; }); render(); });
  ui.prev.addEventListener("click", () => { if (current > 0) { current--; activeTrace = null; setMode("browse"); render(); } });
  ui.next.addEventListener("click", () => { if (current < pack.items.length - 1) { current++; activeTrace = null; setMode("browse"); render(); } });

  ui.overlay.addEventListener("click", (event) => {
    if (event.target.tagName.toLowerCase() === "circle" || currentRecord().submitted) return;
    const p = imagePoint(event), record = currentRecord();
    if (mode === "set-base") {
      clearDownloads(); record.base = p; record.traces.forEach((trace) => { if (trace.points.length) trace.points[0] = { ...p }; }); setMode("browse"); render();
    } else if (mode === "draw") {
      const trace = getTrace(); if (!trace) return message("请先添加并选择一个项目。", "error"); clearDownloads(); trace.points.push(p); persist(); render();
    }
  });

  ui.overlay.addEventListener("pointerdown", (event) => {
    if (currentRecord().submitted || event.target.tagName.toLowerCase() !== "circle") return;
    const pointIndex = Number(event.target.getAttribute("data-point")); if (pointIndex === 0) return;
    drag = { traceId: event.target.getAttribute("data-trace"), pointIndex }; event.target.setPointerCapture(event.pointerId); event.preventDefault();
  });
  ui.overlay.addEventListener("pointermove", (event) => {
    if (!drag) return; const trace = currentRecord().traces.find((t) => t.trace_uuid === drag.traceId); if (!trace) return;
    clearDownloads(); trace.points[drag.pointIndex] = imagePoint(event); persist(); renderOverlay();
  });
  ui.overlay.addEventListener("pointerup", () => { if (drag) { drag = null; render(); } });

  function canonical(value) { if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`; if (value && typeof value === "object") return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`; return JSON.stringify(value); }
  async function sha256(text) {
    if (!crypto.subtle) throw new Error("当前浏览器不支持SHA-256，请改用Edge/Chrome。");
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  }
  function csvCell(value) { const text = value === null || value === undefined ? "" : String(value); return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; }
  function toCsv(headers, rows) { return `\ufeff${[headers.join(","), ...rows.map((row) => headers.map((h) => csvCell(row[h])).join(","))].join("\r\n")}\r\n`; }
  function download(name, content, type) { const blob = new Blob([content], { type }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  function addDownloadLink(name, content, type, label) { const blob = new Blob([content], { type }); const url = URL.createObjectURL(blob); downloadUrls.push(url); const a = document.createElement("a"); a.href = url; a.download = name; a.textContent = label; ui.downloadArea.appendChild(a); }

  async function buildExportBundle(requireComplete = false) {
    const all = pack.items.map((item) => records[item.blind_id]);
    if (requireComplete && all.some((record) => !record.submitted)) throw new Error(`仍有 ${all.filter((record) => !record.submitted).length} 张未提交。`);
    const submitted = all.filter((record) => record.submitted), sessionRows = [], traceRows = [];
    for (const record of submitted) {
      const rawBase = {
        protocol_version: PROTOCOL, package_id: pack.package_id, blind_id: record.blind_id, rater_id: pack.rater_id,
        measurement_round: pack.measurement_round, randomized_order: record.randomized_order, image_alias: record.image_alias,
        plant_submission_uuid: record.plant_submission_uuid, session_status: "submitted", common_base_x_px: record.base.x,
        common_base_y_px: record.base.y, discovered_trace_count: record.traces.length,
        measurable_trace_count: record.traces.filter((t) => t.visibility_status === "measurable").length,
        submission_revision: record.submission_revision, submitted_at_utc: record.submitted_at_utc, notes: record.plant_notes,
      };
      rawBase.raw_export_sha256 = await sha256(canonical(rawBase)); sessionRows.push(rawBase);
      for (const trace of record.traces) {
        const row = {
          protocol_version: PROTOCOL, package_id: pack.package_id, blind_id: record.blind_id, rater_id: pack.rater_id,
          measurement_round: pack.measurement_round, plant_submission_uuid: record.plant_submission_uuid,
          trace_uuid: trace.trace_uuid, gt_leaf_id_postsubmit: trace.gt_leaf_id_postsubmit,
          visibility_status: trace.visibility_status, trace_status: trace.trace_status, basal_transition_status: "marked",
          start_x_px: record.base.x, start_y_px: record.base.y, tip_x_px: trace.tip_xy.x, tip_y_px: trace.tip_xy.y,
          trace_file: `${pack.package_id}_traces.json`, structural_path_length_px: trace.structural_path_length_px,
          shoot_bbox_diagonal_px: pack.items.find((x) => x.blind_id === record.blind_id).bbox_diag_px,
          structural_path_length_bbox_norm: trace.structural_path_length_bbox_norm,
          reference_main_path: trace.reference_main_path ? "yes" : "no",
          divergence_point_x_px: trace.divergence_point_x_px ?? "", divergence_point_y_px: trace.divergence_point_y_px ?? "",
          local_interval_px: trace.branch_local_interval_px ?? "", divergence_angle_deg: trace.divergence_angle_deg ?? "",
          total_turning_angle_deg: trace.total_turning_angle_deg, occlusion_grade: trace.occlusion_grade,
          interpolation_used: trace.interpolation_used, confidence: trace.confidence, evaluation_only_no_training: 1,
          submission_revision: record.submission_revision, notes: trace.notes,
        };
        row.raw_record_sha256 = await sha256(canonical(row)); traceRows.push(row);
      }
    }
    const payload = {
      protocol_version: PROTOCOL, geometry_protocol_version: GEOMETRY, resampling_method: RESAMPLING,
      angle_protocol: ANGLE_PROTOCOL, package_id: pack.package_id, rater_id: pack.rater_id,
      measurement_round: pack.measurement_round, exported_at_utc: new Date().toISOString(), records: submitted,
    };
    payload.export_sha256 = await sha256(canonical(payload));
    return { payload, sessionRows, traceRows };
  }

  ui.exportProgress.addEventListener("click", async () => { try { const bundle = await buildExportBundle(false); download(`${pack.package_id}_progress.json`, JSON.stringify(bundle.payload, null, 2), "application/json"); message("已导出匿名进度备份。", "ok"); } catch (error) { message(error.message, "error"); } });
  ui.exportFinal.addEventListener("click", async () => {
    try {
      const bundle = await buildExportBundle(true);
      const sessionHeaders = ["protocol_version","package_id","blind_id","rater_id","measurement_round","randomized_order","image_alias","plant_submission_uuid","session_status","common_base_x_px","common_base_y_px","discovered_trace_count","measurable_trace_count","submission_revision","submitted_at_utc","raw_export_sha256","notes"];
      const traceHeaders = ["protocol_version","package_id","blind_id","rater_id","measurement_round","plant_submission_uuid","trace_uuid","gt_leaf_id_postsubmit","visibility_status","trace_status","basal_transition_status","start_x_px","start_y_px","tip_x_px","tip_y_px","trace_file","structural_path_length_px","shoot_bbox_diagonal_px","structural_path_length_bbox_norm","reference_main_path","divergence_point_x_px","divergence_point_y_px","local_interval_px","divergence_angle_deg","total_turning_angle_deg","occlusion_grade","interpolation_used","confidence","evaluation_only_no_training","submission_revision","raw_record_sha256","notes"];
      clearDownloads();
      addDownloadLink(`${pack.package_id}_sessions.csv`, toCsv(sessionHeaders, bundle.sessionRows), "text/csv;charset=utf-8", "① 下载 sessions CSV");
      addDownloadLink(`${pack.package_id}_traces.csv`, toCsv(traceHeaders, bundle.traceRows), "text/csv;charset=utf-8", "② 下载 traces CSV");
      addDownloadLink(`${pack.package_id}_traces.json`, JSON.stringify(bundle.payload, null, 2), "application/json", "③ 下载 traces JSON");
      ui.downloadArea.classList.remove("hidden");
      message("全包已完成。请依次点击下方三个链接并一起保存。", "ok");
    } catch (error) { message(error.message, "error"); }
  });

  ui.importInput.addEventListener("change", async (event) => {
    const file = event.target.files[0]; if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (payload.package_id !== pack.package_id || !Array.isArray(payload.records)) throw new Error("备份文件不属于当前测量包");
      for (const record of payload.records) if (records[record.blind_id]) records[record.blind_id] = record;
      persist(); activeTrace = null; render(); message("匿名进度已恢复。", "ok");
    } catch (error) { message(`恢复失败：${error.message}`, "error"); }
    event.target.value = "";
  });

  ui.image.addEventListener("load", renderOverlay);
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (event.key === "ArrowLeft" && current > 0) ui.prev.click();
    if (event.key === "ArrowRight" && current < pack.items.length - 1) ui.next.click();
  });

  window.__GT_DEBUG__ = {
    getState: () => JSON.parse(JSON.stringify({ current, mode, record: currentRecord(), pack: { package_id: pack.package_id, item_count: pack.items.length, training_mode: !!pack.training_mode } })),
    buildExportBundle,
    geometry: { resampleTrace, metrics, chooseMain, divergence },
  };

  render();
})();
