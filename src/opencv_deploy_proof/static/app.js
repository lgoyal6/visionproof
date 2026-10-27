"use strict";
const $ = (id) => document.getElementById(id);
const publicDemo = window.VISIONPROOF_PUBLIC || null;
function resource(url) {
  if (!publicDemo) return url;
  if (!publicDemo.routes[url]) throw new Error("This resource is available in the local workbench.");
  return publicDemo.routes[url];
}
const state = {
  catalog: null,
  run: null,
  image: null,
  pixels: null,
  template: null,
  experiment: 0,
  tab: "inspection",
  guide: -1,
  job: null,
  comparisons: [],
  comparison: null,
};
const fmt = (n) =>
  Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 });
const pct = (n) => (100 * Number(n)).toFixed(1) + "%";
const node = (tag, text, className) => {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
};
function clear(el) {
  el.replaceChildren();
  return el;
}
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
async function api(url, options) {
  if (publicDemo && options) throw new Error("Public replay is read-only.");
  const res = await fetch(resource(url), options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}
function nameOf(run) {
  if (run.pipeline === "pedestrian-yolox")
    return run.provider === "rules"
      ? "Pedestrians, diagnostic control"
      : "Pedestrians, Bedrock search";
  if (run.id.startsWith("nova-guided")) return "Nova 2, diagnostic search";
  if (run.id.startsWith("nova2-proof")) return "Nova 2, original search";
  if (run.id.startsWith("nova-proof"))
    return "Nova Pro, search " + run.id.slice(-2);
  if (run.provider === "rules")
    return run.id.startsWith("local-")
      ? "Local diagnostic control"
      : "Diagnostic control";
  return run.id.startsWith("local-") ? "Bedrock qualification" : run.id;
}
function modelOf(run) {
  return run.provider === "rules"
    ? "Deterministic controller"
    : (run.model || "Bedrock agent")
        .replace("us.amazon.", "")
        .replace("amazon.", "");
}
async function loadCatalog(preferred) {
  const catalog = await api("/api/catalog");
  state.catalog = catalog;
  const list = clear($("run-list"));
  catalog.runs.forEach((run) => {
    const button = node("button", undefined, "run-button");
    button.dataset.run = run.id;
    const title = node("span", undefined, "run-name");
    title.append(
      node("i", undefined, "run-dot" + (run.passed ? "" : " failed")),
      document.createTextNode(nameOf(run)),
    );
    const valid =
      run.speedup > 1.01
        ? fmt(run.speedup) + "× baseline"
        : "Baseline retained";
    button.append(
      title,
      node("span", valid + " / " + run.rejected + " rejected", "run-meta"),
    );
    button.addEventListener("click", () => loadRun(run.id).catch(showError));
    list.append(button);
  });
  $("provider").querySelector("[value=bedrock]").disabled =
    !catalog.capabilities.bedrock;
  await loadRun(preferred || state.run?.id || catalog.default_run);
}
function showError(error) {
  notice(error.message || String(error));
}
async function loadRun(id) {
  notice("");
  const data = await api("/api/run?run=" + encodeURIComponent(id));
  state.run = data;
  state.experiment = 0;
  document
    .querySelectorAll(".run-button")
    .forEach((x) => x.classList.toggle("active", x.dataset.run === id));
  $("run-title").textContent = nameOf(data);
  $("pipeline-title").textContent =
    state.tab === "live"
      ? "OpenCV pipeline / Pedestrian comparison"
      : data.pipeline === "pedestrian-yolox"
      ? "OpenCV pipeline / Pedestrian localization"
      : "OpenCV pipeline / PCB defect localization";
  const datasetLink = document.querySelector(".source-link");
  datasetLink.textContent =
    data.pipeline === "pedestrian-yolox"
      ? "Dataset: Penn-Fudan"
      : "Dataset: DeepPCB";
  datasetLink.href =
    data.pipeline === "pedestrian-yolox"
      ? "https://www.cis.upenn.edu/~jshi/ped_html/"
      : "https://github.com/tangsanli5201/DeepPCB";
  $("template").disabled = data.pipeline === "pedestrian-yolox";
  $("template").checked = false;
  $("run-subtitle").textContent =
    modelOf(data) + " / " + data.test_images + " test images";
  $("run-status").textContent = data.passed
    ? "✓ Qualified for local preview"
    : "Promotion refused";
  $("run-status").classList.toggle("failed", !data.passed);
  $("experiment-count").textContent = data.trace.length;
  $("provider-label").textContent =
    data.provider === "rules"
      ? "Deterministic control, not an LLM"
      : "Real Bedrock experiment trace";
  const link = "/api/export?run=" + encodeURIComponent(id);
  $("download").href = resource(link);
  $("download-report").href = resource(link);
  $("raw-evidence").href = resource("/api/evidence?run=" + encodeURIComponent(id));
  const select = clear($("board"));
  data.images.forEach((im, index) => {
    const option = node(
      "option",
      im.id +
        " / " +
        im.labels.length +
        (im.labels.length === 1 ? " annotation" : " annotations"),
    );
    option.value = index;
    select.append(option);
  });
  renderGate();
  renderExperiments();
  renderReport();
  await loadBoard(0);
  $("loading").hidden = true;
  $("app").hidden = false;
}
function renderGate() {
  const r = state.run,
    req = r.requirements;
  const minRecall = r.baseline_quality.recall - req.max_recall_drop,
    minPrecision = r.baseline_quality.precision - req.max_precision_drop,
    maxP95 = r.baseline_performance.p95_batch_ms * req.max_p95_ratio;
  const checks = [
    [
      "Recall",
      r.quality.recall >= minRecall,
      pct(r.quality.recall),
      "Floor " + pct(minRecall),
    ],
    [
      "Precision",
      r.quality.precision >= minPrecision,
      pct(r.quality.precision),
      "Floor " + pct(minPrecision),
    ],
    [
      "Warm batch p95",
      r.selected_performance.p95_batch_ms <= maxP95,
      fmt(r.selected_performance.p95_batch_ms) + " ms",
      "Ceiling " + fmt(maxP95) + " ms",
    ],
  ];
  const host = clear($("gate-checks"));
  for (const [label, pass, value, floor] of checks) {
    const row = node("div", undefined, "gate-row");
    const left = node("div");
    left.append(
      node("span", pass ? "✓" : "×", "check" + (pass ? "" : " no")),
      document.createTextNode(label),
      node("small", floor),
    );
    row.append(left, node("strong", value));
    host.append(row);
  }
  $("gate-badge").textContent = r.passed ? "Passed" : "Failed";
  $("gate-badge").className = "badge" + (r.passed ? "" : " fail");
  const out = clear($("outcome"));
  [
    ["Baseline speedup", fmt(r.speedup) + "×"],
    ["Versus fixed sweep", r.sweep_ratio.toFixed(2) + "×"],
    ["Rejected candidates", String(r.rejected)],
  ].forEach(([label, value]) => {
    const row = node("div", undefined, "outcome-row");
    row.append(node("span", label), node("strong", value));
    out.append(row);
  });
}
function imageUrl(kind) {
  return (
    "/api/pixels?run=" +
    encodeURIComponent(state.run.id) +
    "&id=" +
    encodeURIComponent(state.image.id) +
    "&kind=" +
    kind
  );
}
function getImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Image could not be loaded"));
    img.src = resource(url);
  });
}
async function loadBoard(index) {
  const images = state.run.images;
  index = (Number(index) + images.length) % images.length;
  state.image = images[index];
  $("board").value = index;
  $("previous-board").disabled = false;
  $("next-board").disabled = false;
  const selectedId = state.image.id;
  const [pixels, template] = await Promise.all([
    getImage(imageUrl("inspection")),
    getImage(imageUrl("template")),
  ]);
  if (state.image.id !== selectedId) return;
  state.pixels = pixels;
  state.template = template;
  renderBoard();
  $("inference-result").textContent = publicDemo
    ? "Recorded detections. Run the local app for fresh inference."
    : "Ready for a fresh measurement.";
  const ctx = $("crop").getContext("2d");
  ctx.clearRect(0, 0, 240, 150);
  $("crop-caption").textContent = "Select an annotation in either image.";
}
function renderBoard() {
  if (!state.pixels) return;
  const im = state.image;
  for (const variant of ["baseline", "selected"]) {
    const canvas = $(variant + "-canvas"),
      ctx = canvas.getContext("2d"),
      result = im[variant];
    canvas.width = state.pixels.width;
    canvas.height = state.pixels.height;
    canvas.style.aspectRatio = canvas.width + " / " + canvas.height;
    ctx.drawImage($("template").checked ? state.template : state.pixels, 0, 0);
    ctx.lineWidth = 2.4;
    const missed = new Set(result.missed_indices);
    if ($("show-predictions").checked) {
      ctx.strokeStyle = "#ffc357";
      ctx.setLineDash([5, 3]);
      for (const box of result.predictions) drawBox(ctx, box, 3);
      ctx.setLineDash([]);
    }
    if ($("show-labels").checked)
      im.labels.forEach((box, i) => {
        ctx.strokeStyle = missed.has(i) ? "#ff6b77" : "#52dec6";
        ctx.lineWidth = missed.has(i) ? 3 : 2;
        drawBox(ctx, box, 0);
      });
    $(variant + "-image-score").textContent =
      result.tp + " found / " + result.fn + " missed";
  }
  const r = state.run;
  $("baseline-batch").textContent =
    fmt(r.baseline_performance.median_batch_ms) + " ms / batch";
  $("selected-batch").textContent =
    fmt(r.selected_performance.median_batch_ms) + " ms / batch";
  $("selected-config-label").textContent =
    (r.pipeline === "pedestrian-yolox"
      ? "Input " + r.config.input_size + " px"
      : "Scale " + r.config.scale) +
    " / " +
    r.config.workers +
    " workers";
  $("infer").disabled = Boolean(publicDemo) || !r.passed;
}
function drawBox(ctx, b, pad) {
  ctx.strokeRect(
    b[0] - pad,
    b[1] - pad,
    b[2] - b[0] + 2 * pad,
    b[3] - b[1] + 2 * pad,
  );
}
function inspectCrop(event, variant) {
  if (!state.pixels) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x =
      ((event.clientX - rect.left) * event.currentTarget.width) / rect.width,
    y = ((event.clientY - rect.top) * event.currentTarget.height) / rect.height;
  let best = 0,
    distance = Infinity;
  state.image.labels.forEach((box, i) => {
    const d = Math.hypot((box[0] + box[2]) / 2 - x, (box[1] + box[3]) / 2 - y);
    if (d < distance) {
      distance = d;
      best = i;
    }
  });
  showCrop(best, variant);
}
function showCrop(index, variant = "selected") {
  const b = state.image.labels[index],
    crop = $("crop"),
    ctx = crop.getContext("2d");
  const size = Math.max(70, b[2] - b[0] + 35, b[3] - b[1] + 35);
  const cx = (b[0] + b[2]) / 2,
    cy = (b[1] + b[3]) / 2;
  ctx.fillStyle = "#152c3b";
  ctx.fillRect(0, 0, crop.width, crop.height);
  ctx.drawImage(
    state.pixels,
    cx - size / 2,
    cy - size / 2,
    size,
    size,
    45,
    0,
    150,
    150,
  );
  ctx.strokeStyle = state.image[variant].missed_indices.includes(index)
    ? "#ff6b77"
    : "#52dec6";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    45 + ((b[0] - cx + size / 2) / size) * 150,
    ((b[1] - cy + size / 2) / size) * 150,
    ((b[2] - b[0]) / size) * 150,
    ((b[3] - b[1]) / size) * 150,
  );
  const missed = state.image[variant].missed_indices.includes(index);
  $("crop-caption").textContent =
    "Annotation " +
    (index + 1) +
    ": " +
    (missed ? "missed" : "matched") +
    " by " +
    variant +
    ". " +
    (b[2] - b[0]) +
    " × " +
    (b[3] - b[1]) +
    " pixels.";
}
function setTab(tab, load = true) {
  state.tab = tab;
  document.querySelector(".run-strip").hidden = tab === "live";
  $("run-status").hidden = tab === "live";
  $("raw-evidence").hidden = tab === "live";
  $("new-run").hidden = Boolean(publicDemo) || tab === "live";
  $("pipeline-title").textContent = tab === "live" ? "OpenCV pipeline / Pedestrian comparison"
    : state.run?.pipeline === "pedestrian-yolox" ? "OpenCV pipeline / Pedestrian localization"
    : "OpenCV pipeline / PCB defect localization";
  document
    .querySelectorAll(".tab-panel")
    .forEach((x) => (x.hidden = x.id !== tab));
  document
    .querySelectorAll("[data-tab]")
    .forEach((x) =>
      x.setAttribute("aria-selected", String(x.dataset.tab === tab)),
    );
  if (tab === "deployment") loadCloud().catch(showError);
  if (tab === "replication" && load) loadComparisons().catch(showError);
  document.dispatchEvent(new CustomEvent("visionproof:tab", { detail: tab }));
}
function renderExperiments() {
  const r = state.run,
    host = clear($("experiment-list"));
  r.trace.forEach((entry, index) => {
    const button = node(
      "button",
      undefined,
      "experiment-row" + (index === state.experiment ? " active" : ""),
    );
    const copy = node("div", undefined, "experiment-copy");
    const changed = Object.entries(entry.decision.config || {}).filter(
      ([k, v]) => v !== r.baseline_config[k],
    );
    copy.append(
      node(
        "strong",
        changed.map(([k, v]) => k.replaceAll("_", " ") + " " + v).join(", ") ||
          "Baseline configuration",
      ),
      node("span", entry.decision.hypothesis),
    );
    button.append(
      node("span", String(index + 1), "experiment-index"),
      copy,
      node(
        "span",
        fmt(entry.performance.median_batch_ms) + " ms",
        "experiment-speed",
      ),
      node(
        "span",
        entry.gate.passed ? "Passed" : "Rejected",
        "badge" + (entry.gate.passed ? "" : " fail"),
      ),
    );
    button.addEventListener("click", () => {
      state.experiment = index;
      renderExperiments();
    });
    host.append(button);
  });
  renderExperimentDetail(r.trace[state.experiment]);
  drawChart();
}
function renderExperimentDetail(entry) {
  const host = clear($("experiment-detail"));
  if (!entry) {
    host.append(node("h3", "No proposals recorded"));
    return;
  }
  host.append(
    node(
      "span",
      entry.gate.passed
        ? "Independent gate passed"
        : "Independent gate rejected",
      "badge" + (entry.gate.passed ? "" : " fail"),
    ),
    node("h3", "Experiment " + entry.index),
    node("p", entry.decision.hypothesis),
  );
  const reasoning = entry.decision.experiment;
  if (reasoning) {
    host.append(
      node("h3", "Falsification condition"),
      node("p", reasoning.falsification),
    );
  }
  const diff = node("div", undefined, "config-diff");
  Object.entries(entry.decision.config || {}).forEach(([k, v]) => {
    const row = node("div");
    row.append(node("span", k.replaceAll("_", " ")), node("strong", String(v)));
    diff.append(row);
  });
  host.append(diff);
  const diag = entry.diagnostics;
  if (diag) {
    host.append(node("h3", "Paired failure evidence"));
    const counts = node(
      "p",
      diag.newly_missed_count +
        " previously matched objects lost; " +
        diag.recovered_count +
        " recovered.",
    );
    host.append(counts);
    if (diag.newly_missed_examples?.length) {
      const samples = node("div", undefined, "failure-samples");
      host.append(samples);
      diag.newly_missed_examples
        .slice(0, 3)
        .forEach((sample) =>
          renderFailureSample(samples, sample, state.run.id),
        );
    }
    if (diag.size_buckets) {
      const list = node("ul");
      Object.entries(diag.size_buckets).forEach(([size, n]) =>
        list.append(
          node(
            "li",
            size +
              ": " +
              n.candidate_missed +
              " missed / " +
              n.total +
              " labels",
          ),
        ),
      );
      host.append(list);
    }
  }
  host.append(node("h3", "Observed stage costs"));
  const total = Object.values(entry.stages_ms).reduce((a, b) => a + b, 0) || 1;
  Object.entries(entry.stages_ms).forEach(([key, value]) => {
    const line = node("div", undefined, "stage");
    const track = node("div", undefined, "stage-track"),
      fill = node("div", undefined, "stage-fill");
    fill.style.width = (100 * value) / total + "%";
    track.append(fill);
    line.append(
      node("span", key.replaceAll("_", " ")),
      node("span", fmt(value) + " ms"),
      track,
    );
    host.append(line);
  });
  const failures = Object.entries(entry.gate.checks)
    .filter(([, v]) => !v)
    .map(([k]) => k);
  host.append(
    node(
      "p",
      failures.length
        ? "Failed checks: " + failures.join(", ") + "."
        : "All declared requirements passed.",
      "fine-print",
    ),
  );
}
function svg(tag, attrs, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
  if (text !== undefined) el.textContent = text;
  return el;
}
function drawChart() {
  const chart = clear($("tradeoff")),
    r = state.run;
  const observations = r.trace;
  const xs = observations.map((x) => x.performance.images_per_second);
  xs.push(r.development_performance.images_per_second);
  const maxX = Math.max(...xs) * 1.18;
  const ys = observations.map((x) => x.quality.recall);
  const minY = Math.max(
      0,
      Math.min(...ys, r.development_quality.recall) - 0.08,
    ),
    maxY = Math.min(1, Math.max(...ys, r.development_quality.recall) + 0.08);
  const x = (n) => 70 + (n / maxX) * 650,
    y = (n) => 214 - ((n - minY) / (maxY - minY || 1)) * 175;
  for (let i = 0; i <= 4; i++) {
    const value = minY + ((maxY - minY) * i) / 4;
    chart.append(
      svg("line", {
        x1: 70,
        x2: 724,
        y1: y(value),
        y2: y(value),
        stroke: "#e6edf2",
      }),
      svg(
        "text",
        {
          x: 58,
          y: y(value) + 4,
          "text-anchor": "end",
          fill: "#607788",
          "font-size": 11,
        },
        pct(value),
      ),
    );
  }
  for (let i = 0; i <= 4; i++) {
    const value = (maxX * i) / 4;
    chart.append(
      svg(
        "text",
        {
          x: x(value),
          y: 235,
          "text-anchor": "middle",
          fill: "#607788",
          "font-size": 11,
        },
        Math.round(value),
      ),
    );
  }
  chart.append(
    svg(
      "text",
      {
        x: 400,
        y: 256,
        "text-anchor": "middle",
        fill: "#607788",
        "font-size": 11,
      },
      "Images per second, warm batch",
    ),
  );
  const devRecall = r.development_quality.recall;
  const acceptance = devRecall - r.requirements.max_recall_drop;
  if (acceptance >= minY && acceptance <= maxY) {
    chart.append(
      svg("line", {
        x1: 70,
        x2: 724,
        y1: y(acceptance),
        y2: y(acceptance),
        stroke: "#a66f3a",
        "stroke-dasharray": "5 4",
      }),
      svg(
        "text",
        {
          x: 718,
          y: y(acceptance) - 6,
          "text-anchor": "end",
          fill: "#966534",
          "font-size": 10,
        },
        "Recall floor " + pct(acceptance),
      ),
    );
  }
  chart.append(
    svg("circle", {
      cx: x(r.development_performance.images_per_second),
      cy: y(devRecall),
      r: 5,
      fill: "#6a7f8d",
    }),
    svg(
      "text",
      {
        x: x(r.development_performance.images_per_second) + 10,
        y: y(devRecall) - 9,
        fill: "#6a7f8d",
        "font-size": 10,
      },
      "Baseline",
    ),
  );
  observations.forEach((p, i) => {
    const color = p.gate.passed ? "#087c79" : "#b84c52";
    chart.append(
      svg("circle", {
        cx: x(p.performance.images_per_second),
        cy: y(p.quality.recall),
        r: i === state.experiment ? 9 : 6,
        fill: color,
        stroke: "white",
        "stroke-width": 2,
      }),
    );
    const label = svg(
      "text",
      {
        x: x(p.performance.images_per_second) + 12,
        y: y(p.quality.recall) + 4,
        fill: color,
        "font-size": 12,
      },
      String(i + 1),
    );
    chart.append(label);
  });
}
function studyName(study) {
  if (study.id.startsWith("live-check-"))
    return "Local dataset check / " + study.id.slice(-6);
  if (study.id.startsWith("ci-pcb-blocked"))
    return "PCB: a faster change blocked by CI";
  if (study.id.startsWith("ci-pcb-passed"))
    return "PCB: a qualified change passes CI";
  if (study.pipeline === "pedestrian-yolox")
    return "Pedestrians: randomized replication";
  return "PCB: reserved-board replication";
}
async function loadComparisons() {
  const data = await api("/api/comparisons");
  state.comparisons = data.comparisons;
  const select = clear($("comparison-study"));
  data.comparisons.forEach((study) => {
    const option = node("option", studyName(study));
    option.value = study.id;
    select.append(option);
  });
  $("comparison-empty").hidden = data.comparisons.length > 0;
  $("comparison-content").hidden = !data.comparisons.length;
  $("comparison-download").hidden = !data.comparisons.length;
  if (!data.comparisons.length) return;
  const previous = state.comparison?.id;
  select.value = data.comparisons.some((s) => s.id === previous)
    ? previous
    : data.comparisons[0].id;
  renderComparison(select.value);
}
function renderComparison(id) {
  const previousCandidate =
    state.comparison?.id === id ? $("comparison-candidate").value : null;
  const study = state.comparisons.find((s) => s.id === id);
  state.comparison = study;
  $("comparison-scope").textContent =
    `${study.image_count} test images / ${study.group_count} source groups / ${study.rounds} randomized blocks / ${study.repeats} timed batches per configuration per block. Recorded local evidence.`;
  $("comparison-download").href =
    resource("/api/comparison-export?comparison=" + encodeURIComponent(id));
  const tbody = clear($("comparison-table").querySelector("tbody"));
  const select = clear($("comparison-candidate"));
  study.rows.forEach((row) => {
    const tr = node("tr");
    [
      row.name,
      row.gate.passed ? "Passed" : "BLOCKED",
      pct(row.quality.recall),
      pct(row.quality.precision),
      fmt(row.median_batch_ms) + " ms",
      row.lost_count + " / " + row.recovered_count,
    ].forEach((value, i) => {
      const td = node("td", value);
      if (i === 1)
        td.className = row.gate.passed ? "comparison-pass" : "comparison-fail";
      tr.append(td);
    });
    tbody.append(tr);
    if (row.name !== "baseline") {
      const option = node("option", row.name);
      option.value = row.name;
      select.append(option);
    }
  });
  if (
    previousCandidate &&
    study.rows.some((row) => row.name === previousCandidate)
  )
    select.value = previousCandidate;
  const limits = clear($("comparison-limits"));
  study.limitations.forEach((t) => limits.append(node("li", t)));
  const pairwise = study.pairwise.filter(
    (p) => p.numerator !== "baseline" && p.denominator !== "baseline",
  );
  $("comparison-pairwise").textContent = pairwise
    .map((p) => {
      const v = p.speedup_of_denominator;
      return (
        `${p.denominator} versus ${p.numerator}: ${v.median.toFixed(2)}× [${v.low.toFixed(2)}, ${v.high.toFixed(2)}]. ` +
        (v.low <= 1 && v.high >= 1
          ? "The timing interval includes parity."
          : "Timing evidence applies only to these frozen configurations.")
      );
    })
    .join(" ");
  const dataset =
    study.pipeline === "pedestrian-yolox" ? "pedestrian-data" : "pcb-data";
  const config = study.pipeline === "pedestrian-yolox" ? "pedestrians" : "pcb";
  $("ci-command").textContent = String.raw`vision-proof check \
  --manifest ${dataset}/manifest.json \
  --candidate-config configs/${config}.json \
  --rounds 5 --repeats 5 \
  --out qualification-report`;
  const chart = clear($("comparison-plot"));
  const max =
    Math.max(1.2, ...study.rows.map((r) => r.paired_speedup.high)) * 1.15;
  const x = (n) => 155 + (590 * n) / max;
  const height = 40 + study.rows.length * 44;
  chart.setAttribute("viewBox", `0 0 800 ${height}`);
  chart.append(
    svg("line", {
      x1: x(1),
      x2: x(1),
      y1: 10,
      y2: height - 25,
      stroke: "#889ba9",
      "stroke-dasharray": "4 4",
    }),
  );
  study.rows.forEach((row, i) => {
    const y = 28 + i * 44,
      speed = row.paired_speedup,
      color = row.gate.passed ? "#087c79" : "#b84c52";
    chart.append(
      svg(
        "text",
        { x: 12, y: y + 4, fill: "#193348", "font-size": 13 },
        row.name,
      ),
      svg("line", {
        x1: x(speed.low),
        x2: x(speed.high),
        y1: y,
        y2: y,
        stroke: color,
        "stroke-width": 3,
      }),
      svg("circle", { cx: x(speed.median), cy: y, r: 6, fill: color }),
      svg(
        "text",
        {
          x: Math.min(750, x(speed.high) + 12),
          y: y + 4,
          fill: color,
          "font-size": 12,
        },
        speed.median.toFixed(2) + "×",
      ),
    );
  });
  chart.append(
    svg(
      "text",
      {
        x: x(1),
        y: height - 5,
        fill: "#607788",
        "font-size": 11,
        "text-anchor": "middle",
      },
      "Baseline 1×",
    ),
  );
  renderComparisonCandidate().catch(showError);
}
async function renderComparisonCandidate() {
  const study = state.comparison,
    row = study.rows.find((r) => r.name === $("comparison-candidate").value);
  if (!row) return;
  const failures = Object.entries(row.gate.checks)
    .filter(([, pass]) => !pass)
    .map(([name]) => name.replaceAll("_", " "));
  $("comparison-checks").textContent = row.gate.passed
    ? "Every aggregate check passed. " +
      row.round_gates.filter((g) => g.passed).length +
      "/" +
      study.rounds +
      " individual round gates passed."
    : "Blocked by: " +
      failures.join(", ") +
      ". Faster processing cannot override lost detections.";
  const groups = clear($("comparison-groups"));
  groups.append(node("h3", "Quality by source group"));
  Object.entries(row.by_group).forEach(([name, q]) =>
    groups.append(
      node(
        "p",
        `${name}: recall ${pct(q.recall)}, precision ${pct(q.precision)}, ${q.tp + q.fn} annotations.`,
      ),
    ),
  );
  const lost = row.lost_examples[0],
    example = lost
      ? row.examples.find((e) => e.id === lost.id)
      : row.examples[0];
  const img = await getImage(
    "/api/comparison-pixels?comparison=" +
      encodeURIComponent(study.id) +
      "&id=" +
      encodeURIComponent(example.id),
  );
  if (
    state.comparison?.id !== study.id ||
    $("comparison-candidate").value !== row.name
  )
    return;
  const canvas = $("comparison-image"),
    ctx = canvas.getContext("2d");
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffc357";
  ctx.setLineDash([5, 3]);
  example.predictions.forEach((box) => drawBox(ctx, box, 0));
  ctx.setLineDash([]);
  example.labels.forEach((box, index) => {
    ctx.strokeStyle = example.missed_indices.includes(index)
      ? "#ff6b77"
      : "#52dec6";
    ctx.lineWidth = 3;
    drawBox(ctx, box, 0);
  });
  $("comparison-caption").textContent =
    example.id +
    ": " +
    example.tp +
    " matched, " +
    example.fn +
    " missed. " +
    (lost
      ? "This image contains a detection the baseline found and this candidate lost. "
      : "First registered example. ") +
    "Teal: matched annotation. Red: missed. Amber: prediction.";
}
$("comparison-study").addEventListener("change", (e) =>
  renderComparison(e.target.value),
);
$("comparison-candidate").addEventListener("change", () =>
  renderComparisonCandidate().catch(showError),
);

async function renderFailureSample(host, sample, runId) {
  const figure = node("figure", undefined, "failure-sample"),
    canvas = node("canvas"),
    caption = node(
      "figcaption",
      "Image " +
        sample.image_id +
        " / " +
        sample.width +
        " × " +
        sample.height +
        " px",
    );
  canvas.width = 160;
  canvas.height = 105;
  figure.append(canvas, caption);
  host.append(figure);
  try {
    const img = await getImage(
      "/api/development-pixels?run=" +
        encodeURIComponent(runId) +
        "&id=" +
        encodeURIComponent(sample.image_id),
    );
    if (!figure.isConnected) return;
    const b = sample.box,
      size = Math.max(65, b[2] - b[0] + 30, b[3] - b[1] + 30),
      cx = (b[0] + b[2]) / 2,
      cy = (b[1] + b[3]) / 2,
      ctx = canvas.getContext("2d");
    ctx.fillStyle = "#163345";
    ctx.fillRect(0, 0, 160, 105);
    ctx.drawImage(
      img,
      cx - size / 2,
      cy - size / 2,
      size,
      size,
      27,
      0,
      105,
      105,
    );
    ctx.strokeStyle = "#ff6b77";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      27 + ((b[0] - cx + size / 2) / size) * 105,
      ((b[1] - cy + size / 2) / size) * 105,
      ((b[2] - b[0]) / size) * 105,
      ((b[3] - b[1]) / size) * 105,
    );
  } catch {
    caption.textContent = "Crop unavailable";
  }
}
async function loadCloud() {
  const c = await api("/api/cloud");
  $("cloud-empty").hidden = c.rows.length > 0;
  $("cloud-content").hidden = !c.rows.length;
  if (!c.rows.length) return;
  const max = Math.max(...c.rows.map((r) => r.selected_batch_ms));
  const bars = clear($("cloud-bars")),
    tbody = clear($("cloud-table").querySelector("tbody"));
  const names = {
    "arm-pip": "Graviton / pip",
    "x86-pip": "x86 / pip",
    cool: "Graviton / COOL",
  };
  c.rows.forEach((row) => {
    const card = node("div", undefined, "cloud-bar-card"),
      track = node("div", undefined, "cloud-bar-track"),
      fill = node("div", undefined, "cloud-bar-fill");
    fill.style.width = (100 * row.selected_batch_ms) / max + "%";
    track.append(fill);
    card.append(
      node("h3", names[row.target]),
      node("p", row.instance_type + " / 80 test images"),
      node("strong", fmt(row.selected_batch_ms) + " ms"),
      track,
    );
    bars.append(card);
    const tr = node("tr");
    [
      names[row.target],
      row.instance_type,
      row.environment.opencv + " / " + row.environment.numpy,
      fmt(row.selected_batch_ms) + " ms",
      pct(row.quality.recall) + " / " + pct(row.quality.precision),
      "$" + row.hourly_usd.toFixed(5) + "/hour",
      row.gate.passed ? "Passed" : "Failed",
    ].forEach((t) => tr.append(node("td", t)));
    tbody.append(tr);
  });
  const arm = c.rows.find((r) => r.target === "arm-pip"),
    cool = c.rows.find((r) => r.target === "cool");
  $("cloud-takeaway").textContent =
    arm && cool
      ? "COOL recorded " +
        pct(1 - cool.selected_batch_ms / arm.selected_batch_ms) +
        " lower warm batch latency than pip on the same instance type, with identical scored detections in this sample. Different builds and one run limit attribution."
      : "Measured worker results, subject to the scope below.";
  const limits = clear($("cloud-limits"));
  c.limitations.forEach((t) => limits.append(node("li", t)));
  const cleanup = clear($("cleanup"));
  if (c.cleanup) {
    [
      ["Subscription", c.cleanup.subscription],
      [
        "Workers",
        c.cleanup.instances.every((i) => i.state === "terminated")
          ? "All terminated"
          : "Review required",
      ],
      [
        "SSH keys and security groups",
        c.cleanup.key_pairs.length + c.cleanup.security_groups.length === 0
          ? "Removed"
          : "Review required",
      ],
    ].forEach(([key, value]) => {
      const line = node("div", undefined, "cleanup-row");
      line.append(node("span", "✓"), node("span", key + ": " + value));
      cleanup.append(line);
    });
    cleanup.append(
      node(
        "p",
        "Verified " + new Date(c.cleanup.checked_utc).toLocaleString(),
        "fine-print",
      ),
    );
  } else cleanup.append(node("p", "No cleanup record registered."));
}
function renderReport() {
  const r = state.run;
  $("config-json").textContent = JSON.stringify(r.config, null, 2);
  const host = clear($("report-metrics"));
  [
    ["Baseline recall", pct(r.baseline_quality.recall)],
    ["Selected recall", pct(r.quality.recall)],
    ["Selected precision", pct(r.quality.precision)],
    ["Fixed sweep precision", pct(r.verdict.sweep_test_quality.precision)],
    ["Speed versus fixed sweep", r.sweep_ratio.toFixed(3) + "×"],
  ].forEach(([key, value]) => {
    const line = node("div", undefined, "gate-row");
    line.append(node("span", key), node("strong", value));
    host.append(line);
  });
  const provenance = clear($("provenance"));
  [
    ["Planner", modelOf(r)],
    ["OpenCV", r.environment.opencv],
    ["NumPy", r.environment.numpy],
    ["Architecture", r.environment.machine],
    ["Test image count", r.test_images],
    ["Dataset digest", r.manifest_sha256],
    ["Verdict digest", r.sha256],
  ].forEach(([key, value]) => {
    provenance.append(node("dt", key), node("dd", String(value)));
  });
  const limits = clear($("limitations"));
  [
    ...r.protocol.limitations,
    ...r.verdict.limitations.filter((x) => !x.startsWith("Rules provider")),
    "The simple sweep remains a strong control. This run does not establish LLM search superiority.",
  ].forEach((t) => limits.append(node("li", t)));
  const journal = clear($("journal"));
  $("journal-label").textContent = publicDemo
    ? "Source event references, sanitized replay"
    : r.events.length
    ? r.events.length + " hash-linked events verified"
    : "Legacy run, no event journal";
  r.events.forEach((event) => {
    const li = node("li", event.kind.replaceAll("_", " "));
    li.append(
      node(
        "span",
        new Date(event.time).toLocaleTimeString() +
          " / " +
          event.sha256.slice(0, 12),
      ),
    );
    journal.append(li);
  });
}
async function infer() {
  const b = $("infer");
  b.disabled = true;
  $("inference-result").textContent = "Running the frozen detector…";
  try {
    const answer = await api(
      "/api/inspect?run=" +
        encodeURIComponent(state.run.id) +
        "&id=" +
        encodeURIComponent(state.image.id),
    );
    $("inference-result").textContent =
      answer.quality.tp +
      " matched, " +
      answer.quality.fn +
      " missed. " +
      fmt(answer.wall_ms) +
      " ms local inference.";
    state.image.selected.predictions = answer.boxes;
    state.image.selected.missed_indices = answer.quality.missed_indices;
    state.image.selected.tp = answer.quality.tp;
    state.image.selected.fn = answer.quality.fn;
    renderBoard();
  } catch (e) {
    $("inference-result").textContent = e.message;
  } finally {
    b.disabled = false;
  }
}
const guideSteps = [
  [
    "Faster can mean blind",
    "A reduced-resolution PCB change lost 559 detections the baseline found. The recorded CI check blocked it despite a small speed improvement. Inspect the missed annotation below.",
    "replication",
    "ci-pcb-blocked-01",
  ],
  [
    "Make the passing change prove itself",
    "Four workers preserved the baseline detections and measured 3.56x faster across local paired blocks. That recorded CI command passed. This is evidence for this workload, not a universal speed promise.",
    "replication",
    "ci-pcb-passed-01",
  ],
  [
    "A contract before a speed claim",
    "This saved qualification tested 64 new images. The gate permits at most two percentage points of recall or precision loss.",
    "inspection",
  ],
  [
    "Fast changes can erase defects",
    "The first proposal resized inspection pixels. Its missed defects failed independent evaluation. This is recorded experiment playback.",
    "experiments",
    0,
  ],
  [
    "Follow the evidence back to full resolution",
    "The agent returned to the baseline and tested image-level concurrency. The evaluator accepted the resulting quality.",
    "experiments",
    2,
  ],
  [
    "Inspect the result, not just the number",
    "The selected configuration passed test evaluation. The sweep was about as fast and had better precision; no agent-superiority claim follows.",
    "inspection",
  ],
  [
    "Cloud results come with receipts",
    "These earlier AWS runs include pip on x86 and Graviton, plus an entitled COOL image. Different library builds limit attribution.",
    "deployment",
  ],
  [
    "Take the proof with you",
    "Export source, configuration, input manifest and measured outputs. AWS workers are terminated and the COOL subscription is cancelled.",
    "report",
  ],
];
async function guideStep() {
  const step = guideSteps[state.guide];
  if (!step) {
    $("guide").hidden = true;
    state.guide = -1;
    return;
  }
  $("guide").hidden = false;
  $("guide-title").textContent = step[0];
  $("guide-text").textContent = step[1];
  $("guide-count").textContent = state.guide + 1 + " / " + guideSteps.length;
  $("guide-next").textContent =
    state.guide === guideSteps.length - 1 ? "Finish" : "Next";
  setTab(step[2], step[2] !== "replication");
  if (step[2] === "replication") {
    await loadComparisons();
    const study = state.comparisons.find((s) => s.id === step[3]);
    if (study) {
      $("comparison-study").value = study.id;
      renderComparison(study.id);
    } else {
      $("guide-text").textContent = "This local installation has no recorded regression study yet. Use the check command to generate your own passing and blocked reports.";
    }
  } else if (step[3] !== undefined) {
    state.experiment = Math.min(step[3], state.run.trace.length - 1);
    renderExperiments();
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}
async function startGuide() {
  const target =
    state.catalog.runs.find((r) => r.id === "nova-guided-01") ||
    state.catalog.runs.find(
      (r) => r.provider === "bedrock" && r.speedup > 1.01,
    );
  if (target && target.id !== state.run.id) await loadRun(target.id);
  state.guide = 0;
  await guideStep();
}
async function pollJob() {
  if (!state.job) return;
  try {
    const job = await api("/api/jobs/" + state.job);
    $("job-status").textContent =
      job.status === "failed"
        ? "Failed: " + job.error
        : job.status === "completed"
          ? "Independent qualification finished."
          : "Measuring " + job.provider + " search and fixed sweep…";
    const events = clear($("job-events"));
    job.events.slice(-7).forEach((e) => {
      events.append(
        node(
          "li",
          e.kind.replaceAll("_", " ") +
            (e.payload.experiment ? " " + e.payload.experiment : "") +
            (e.kind === "measured"
              ? " / " + (e.payload.gate.passed ? "passed" : "rejected")
              : ""),
        ),
      );
    });
    if (job.status === "running") {
      setTimeout(pollJob, 1800);
    } else {
      $("job-panel").classList.add("complete");
      $("job-heading").textContent =
        job.status === "completed"
          ? "Qualification complete"
          : "Qualification failed";
      $("open-job").hidden = job.status !== "completed";
      $("open-job").onclick = async () => {
        await loadCatalog(job.id);
        $("job-panel").hidden = true;
        setTab("experiments");
      };
      $("start-job").disabled = false;
      $("infer").disabled = false;
    }
  } catch (e) {
    $("job-status").textContent = e.message;
    $("start-job").disabled = false;
  }
}
document
  .querySelectorAll("[data-tab]")
  .forEach((x) => x.addEventListener("click", () => setTab(x.dataset.tab)));
$("board").addEventListener("change", (e) =>
  loadBoard(e.target.value).catch(showError),
);
$("previous-board").onclick = () =>
  loadBoard(Number($("board").value) - 1).catch(showError);
$("next-board").onclick = () =>
  loadBoard(Number($("board").value) + 1).catch(showError);
["show-predictions", "show-labels", "template"].forEach((id) =>
  $(id).addEventListener("change", renderBoard),
);
["baseline", "selected"].forEach((variant) => {
  $(variant + "-canvas").addEventListener("click", (e) =>
    inspectCrop(e, variant),
  );
  $(variant + "-canvas").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      showCrop(0, variant);
    }
  });
});
$("infer").onclick = infer;
$("refresh").onclick = () => loadCatalog().catch(showError);
$("guide-toggle").onclick = () => startGuide().catch(showError);
$("guide-next").onclick = () => {
  state.guide++;
  guideStep().catch(showError);
};
$("guide-close").onclick = () => {
  $("guide").hidden = true;
  state.guide = -1;
};
$("new-run").onclick = () => $("qualification-dialog").showModal();
$("close-dialog").onclick = () => $("qualification-dialog").close();
$("hide-job").onclick = () => ($("job-panel").hidden = true);
$("qualification-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("start-job").disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run: state.run.id,
        provider: $("provider").value,
        budget: Number($("budget").value),
      }),
    });
    state.job = job.id;
    $("qualification-dialog").close();
    $("job-panel").hidden = false;
    $("job-panel").classList.remove("complete");
    $("job-heading").textContent = "Qualification running";
    $("open-job").hidden = true;
    $("infer").disabled = true;
    pollJob();
  } catch (error) {
    showError(error);
    $("start-job").disabled = false;
    $("qualification-dialog").close();
  }
});
if (publicDemo) {
  document.querySelector(".connection").textContent = "Read-only replay";
  $("guide-toggle").hidden = true;
  $("new-run").disabled = true;
  $("new-run").hidden = true;
  $("infer").disabled = true;
  document.querySelector(".inference-strip").hidden = true;
  $("download").textContent = "Summary JSON";
  $("download-report").textContent = "Download results";
  $("comparison-download").textContent = "Download study summary";
  $("raw-evidence").textContent = "Result details";
  const banner = node("p", "Recorded results. Live evaluation runs in the local app. ", "notice");
  const about = node("a", "Architecture and results");
  about.href = "/about.html";
  banner.append(about);
  document.querySelector(".topbar").after(banner);
}
loadCatalog().then(() => {
  if (!publicDemo && location.hash === "#live") setTab("live");
}).catch((error) => {
  $("loading").textContent =
    "The workbench could not load verified evidence. " + error.message;
  showError(error);
});
