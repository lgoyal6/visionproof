"use strict";
(() => {
  const el = (id) => document.getElementById(id);
  if (window.VISIONPROOF_PUBLIC) {
    el("live-tab").hidden = true;
    return;
  }
  let loaded = false, currentImage = null, frameData = null, activeJob = null;
  let stream = null, cameraGeneration = 0, imageGeneration = 0, timer = null, pending = null;
  const number = (value) => Number(value).toFixed(1);
  function message(text) { el("lab-status").textContent = text; }
  function evaluationControls(busy) {
    for (const id of ["lab-import", "lab-source", "lab-dataset", "lab-image", "lab-upload", "lab-compare",
      "lab-base-size", "lab-base-confidence", "lab-base-workers", "lab-candidate-size", "lab-candidate-confidence", "lab-candidate-workers",
      "lab-recall", "lab-precision", "lab-latency"]) el(id).disabled = busy;
    el("lab-evaluate").disabled = busy || el("lab-source").value !== "dataset";
    el("lab-compare").disabled = busy || !currentImage;
  }
  function error(value) {
    el("lab-error").textContent = value?.message || String(value);
    el("lab-error").hidden = false;
  }
  async function request(path, body, signal) {
    const response = await fetch(path, body === undefined ? { signal } : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    return data;
  }
  function config(prefix) {
    return { input_size: Number(el(prefix + "-size").value), confidence: Number(el(prefix + "-confidence").value),
      workers: Number(el(prefix + "-workers").value), threads: 1 };
  }
  function image(url) {
    return new Promise((resolve, reject) => {
      const img = new Image(); img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Image could not be loaded")); img.src = url;
    });
  }
  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("File could not be read")); reader.readAsDataURL(file);
    });
  }
  function pixels(source, jpeg = false) {
    const width = source.videoWidth || source.naturalWidth || source.width;
    const height = source.videoHeight || source.naturalHeight || source.height;
    if (!width || !height) throw new Error("Waiting for camera pixels");
    const scale = Math.min(1, 1280 / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale)); canvas.height = Math.max(1, Math.round(height * scale));
    canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL(jpeg ? "image/jpeg" : "image/png", .9);
  }
  function draw(prefix, source, result, labels) {
    const canvas = el(prefix + "-canvas"), ctx = canvas.getContext("2d");
    canvas.width = source.naturalWidth || source.width; canvas.height = source.naturalHeight || source.height;
    ctx.drawImage(source, 0, 0);
    ctx.lineWidth = Math.max(2, canvas.width / 300);
    ctx.strokeStyle = "#ffc357";
    for (const box of result?.boxes || []) ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    const missed = new Set(result?.quality?.missed_indices || []);
    if (result && labels) labels.forEach((box, index) => {
      ctx.strokeStyle = missed.has(index) ? "#ff6b77" : "#52dec6";
      ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    });
    el(prefix + "-score").textContent = result
      ? `${result.config.input_size} px, confidence ${result.config.confidence}. ${result.boxes.length} predictions. ` +
        (result.quality ? `${result.quality.tp} matched, ${result.quality.fn} missed, ${result.quality.fp} false positives. ` : "No accuracy labels. ") +
        `${number(result.warm_image_ms)} ms warm inference.`
      : "Ready to compare.";
  }
  function clearPreview() {
    for (const prefix of ["lab-base", "lab-candidate"]) {
      const canvas = el(prefix + "-canvas"); canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      el(prefix + "-score").textContent = "No measurement yet.";
    }
    el("lab-delta").textContent = "Compare an image to inspect detection differences.";
  }
  async function loadImage() {
    const generation = ++imageGeneration;
    frameData = null; currentImage = null;
    el("lab-compare").disabled = true; message("Loading image…");
    clearPreview();
    const dataset = el("lab-dataset").value, id = el("lab-image").value;
    if (!dataset || !id) return;
    const img = await image(`/api/lab/pixels?dataset=${encodeURIComponent(dataset)}&id=${encodeURIComponent(id)}`);
    if (generation !== imageGeneration) return;
    currentImage = img;
    el("lab-compare").disabled = Boolean(activeJob);
    for (const prefix of ["lab-base", "lab-candidate"]) draw(prefix, img);
    message("Ready. Compare these configurations on this labeled image.");
  }
  async function loadImages() {
    const dataset = el("lab-dataset").value;
    currentImage = null; ++imageGeneration; el("lab-compare").disabled = true;
    clearPreview(); message("Loading dataset images…");
    el("lab-image").replaceChildren();
    if (!dataset) { message("Prepare a pedestrian qualification to enable live comparison."); return; }
    const data = await request("/api/lab/images?dataset=" + encodeURIComponent(dataset));
    if (dataset !== el("lab-dataset").value) return;
    data.images.forEach((row) => {
      const option = document.createElement("option"); option.value = row.id;
      option.textContent = `${row.id} / ${row.labels.length} labeled people`; el("lab-image").append(option);
    });
    await loadImage();
  }
  async function refresh(preferred) {
    const data = await request("/api/lab/catalog");
    const previous = preferred || el("lab-dataset").value;
    el("lab-dataset").replaceChildren();
    data.datasets.forEach((dataset) => {
      const option = document.createElement("option"); option.value = dataset.id;
      option.textContent = `${dataset.name} / ${dataset.image_count} images`; el("lab-dataset").append(option);
    });
    if (data.datasets.some((d) => d.id === previous)) el("lab-dataset").value = previous;
    else if (data.datasets.some((d) => d.id === "yolox-control-01")) el("lab-dataset").value = "yolox-control-01";
    await loadImages();
  }
  async function compareImage(dataURL, generation) {
    el("lab-error").hidden = true;
    if (!currentImage && !dataURL) throw new Error("Choose an image first");
    const imageVersion = imageGeneration;
    const src = dataURL ? await image(dataURL) : currentImage;
    const body = { baseline: config("lab-base"), candidate: config("lab-candidate") };
    if (dataURL || frameData) body.frame = (dataURL || frameData).split(",")[1];
    else { body.dataset = el("lab-dataset").value; body.id = el("lab-image").value; }
    pending = new AbortController(); el("lab-compare").disabled = true;
    try {
      message("Running both configurations on the same pixels…");
      const result = await request("/api/lab/compare", body, pending.signal);
      if (generation !== undefined && generation !== cameraGeneration) return;
      if (generation === undefined && imageVersion !== imageGeneration) return;
      draw("lab-base", src, result.baseline, result.labels); draw("lab-candidate", src, result.candidate, result.labels);
      el("lab-delta").textContent = result.labels !== null
        ? `${result.delta.newly_missed_annotations.length} annotations detected by the baseline were missed by the candidate. Amber: predictions. Teal: matched labels. Red: missed labels.`
        : `${result.delta.baseline_only_predictions} baseline predictions have no matching candidate prediction. Without labels this is disagreement, not measured accuracy.`;
      message(stream ? "Camera comparison running. Frames are processed locally and not saved." : "Comparison complete. Dataset evaluation provides the full quality check.");
    } finally { pending = null; el("lab-compare").disabled = !currentImage || Boolean(activeJob); }
  }
  function stopCamera() {
    cameraGeneration++; clearTimeout(timer); pending?.abort();
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null; el("lab-video").srcObject = null; el("lab-video").hidden = true;
    el("lab-camera-start").disabled = false; el("lab-camera-stop").disabled = true;
  }
  async function startCamera() {
    stopCamera(); const generation = cameraGeneration;
    el("lab-error").hidden = true; el("lab-camera-start").disabled = true;
    el("lab-camera-stop").disabled = false;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires localhost or HTTPS and a supported browser");
      message("Waiting for camera permission…");
      const device = el("lab-device").value;
      const acquired = await navigator.mediaDevices.getUserMedia({ audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, ...(device ? { deviceId: { exact: device } } : {}) } });
      if (generation !== cameraGeneration) { acquired.getTracks().forEach((track) => track.stop()); return; }
      stream = acquired; el("lab-video").srcObject = stream;
      stream.getVideoTracks().forEach((track) => track.addEventListener("ended", () => { stopCamera(); message("Camera disconnected."); }));
      el("lab-camera-stop").disabled = false; await el("lab-video").play();
      const selected = stream.getVideoTracks()[0].getSettings().deviceId;
      el("lab-device").replaceChildren();
      for (const camera of (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput")) {
        const option = document.createElement("option"); option.value = camera.deviceId; option.textContent = camera.label || "Camera"; el("lab-device").append(option);
      }
      if (selected) el("lab-device").value = selected;
      const tick = async () => {
        if (generation !== cameraGeneration || !stream) return;
        try { await compareImage(pixels(el("lab-video"), true), generation); }
        catch (e) { if (generation !== cameraGeneration) return; stopCamera(); error(e); message("Camera stopped."); return; }
        if (generation === cameraGeneration) timer = setTimeout(tick, 250);
      };
      await tick();
    } catch (e) {
      if (generation !== cameraGeneration) return;
      stopCamera(); error(new Error(e.name === "NotAllowedError" ? "Camera permission was denied. Allow access in your browser and try again." : e.message));
      message("Camera stopped.");
    }
  }
  async function sourceChanged() {
    stopCamera(); ++imageGeneration; currentImage = null; frameData = null;
    el("lab-compare").disabled = true;
    clearPreview();
    const source = el("lab-source").value;
    for (const id of ["lab-dataset-label", "lab-image-label"]) el(id).hidden = source !== "dataset";
    el("lab-upload-label").hidden = source !== "upload";
    el("lab-device-label").hidden = source !== "camera";
    el("lab-compare").hidden = source === "camera";
    el("lab-camera-start").hidden = el("lab-camera-stop").hidden = source !== "camera";
    el("lab-evaluate").disabled = Boolean(activeJob) || source !== "dataset";
    message(source === "camera" ? "Start the camera to compare sampled frames. No frames are saved." : "Choose an image.");
    if (source === "dataset") await loadImage();
  }
  async function importDataset() {
    stopCamera(); el("lab-error").hidden = true;
    const file = el("lab-import-file").files[0];
    if (!file) throw new Error("Choose a dataset ZIP first");
    if (file.size > 32 * 1024 * 1024) throw new Error("ZIP exceeds 32 MiB");
    el("lab-import").disabled = true; el("lab-import-status").textContent = "Validating images and annotations…";
    try {
      const result = await request("/api/lab/import", { name: el("lab-import-name").value, archive: (await readFile(file)).split(",")[1] });
      el("lab-source").value = "dataset"; await sourceChanged(); await refresh(result.id);
      el("lab-import-status").textContent = "Dataset imported. Labels are available for comparison and evaluation.";
    } catch (e) { el("lab-import-status").textContent = "Import failed. Correct the dataset and try again."; throw e; }
    finally { el("lab-import").disabled = false; }
  }
  async function pollEvaluation() {
    if (!activeJob) return;
    try {
      const job = await request("/api/lab/job?id=" + encodeURIComponent(activeJob));
      if (job.status === "running") { setTimeout(pollEvaluation, 1500); return; }
      evaluationControls(false);
      if (job.status === "failed") throw new Error(job.error);
      const candidate = job.report.rows.find((r) => r.name === "candidate"), baseline = job.report.rows.find((r) => r.name === "baseline");
      el("lab-evaluation-status").textContent = candidate.gate.passed ? "Passed all dataset checks." : "Blocked by the dataset checks.";
      const host = el("lab-evaluation-result"); host.replaceChildren();
      const checks = Object.entries(candidate.gate.checks).map(([name, pass]) => `${name.replaceAll("_", " ")}: ${pass ? "passed" : "failed"}`).join("; ");
      for (const text of [`Evaluated baseline: ${baseline.config.input_size} px, ${baseline.config.workers} workers. Candidate: ${candidate.config.input_size} px, ${candidate.config.workers} workers.`, checks, `Recall: ${number(100 * candidate.quality.recall)}% (baseline ${number(100 * baseline.quality.recall)}%). Precision: ${number(100 * candidate.quality.precision)}%.`,
        `${candidate.lost_count} baseline detections lost. Paired batch speedup ${candidate.paired_speedup.median.toFixed(2)}x; 95% interval ${candidate.paired_speedup.low.toFixed(2)}–${candidate.paired_speedup.high.toFixed(2)}x.`,
        "This evaluates these fixed configurations on this labeled test set. Repeated checks do not create an untouched holdout."]) {
        const p = document.createElement("p"); p.textContent = text; host.append(p);
      }
      el("lab-download").href = "/api/comparison-export?comparison=" + encodeURIComponent(activeJob); el("lab-download").hidden = false;
      activeJob = null;
    } catch (e) { activeJob = null; evaluationControls(false); el("lab-evaluation-status").textContent = "Evaluation could not complete."; error(e); }
  }
  async function evaluateDataset() {
    stopCamera(); el("lab-error").hidden = true;
    if (!el("lab-dataset").value) throw new Error("Choose a labeled dataset first");
    for (const id of ["lab-recall", "lab-precision", "lab-latency"]) {
      if (!el(id).value || !el(id).checkValidity()) throw new Error("Enter valid recall, precision and latency limits");
    }
    evaluationControls(true); el("lab-download").hidden = true;
    try {
      const job = await request("/api/lab/evaluate", { dataset: el("lab-dataset").value,
        baseline: config("lab-base"), candidate: config("lab-candidate"), requirements: {
          max_recall_drop: Number(el("lab-recall").value) / 100, max_precision_drop: Number(el("lab-precision").value) / 100,
          max_p95_ratio: 1 + Number(el("lab-latency").value) / 100 } });
      activeJob = job.id; el("lab-evaluation-result").replaceChildren();
      el("lab-evaluation-status").textContent = "Evaluating the frozen configurations. This may take several minutes.";
      pollEvaluation();
    } catch (e) { evaluationControls(false); throw e; }
  }
  const handle = (fn) => () => Promise.resolve().then(fn).catch((e) => { if (e.name !== "AbortError") error(e); });
  document.addEventListener("visionproof:tab", (event) => {
    if (event.detail !== "live") { stopCamera(); return; }
    if (!loaded) { loaded = true; refresh().catch((e) => { loaded = false; error(e); }); }
  });
  el("lab-source").onchange = handle(sourceChanged); el("lab-dataset").onchange = handle(loadImages);
  el("lab-image").onchange = handle(loadImage);
  el("lab-compare").onclick = handle(() => compareImage());
  el("lab-camera-start").onclick = handle(startCamera);
  el("lab-camera-stop").onclick = () => { stopCamera(); message("Camera stopped. No frames were saved."); };
  el("lab-device").onchange = () => { stopCamera(); message("Camera changed. Start it when ready."); };
  el("lab-import").onclick = handle(importDataset); el("lab-evaluate").onclick = handle(evaluateDataset);
  el("lab-upload").onchange = handle(async () => {
    const generation = ++imageGeneration;
    currentImage = null; frameData = null; el("lab-compare").disabled = true;
    const file = el("lab-upload").files[0]; if (!file) return;
    if (file.size > 8 * 1024 * 1024) throw new Error("Image exceeds 8 MiB");
    const normalized = pixels(await image(await readFile(file)));
    if (generation !== imageGeneration) return;
    const img = await image(normalized);
    if (generation !== imageGeneration) return;
    frameData = normalized; currentImage = img;
    el("lab-compare").disabled = Boolean(activeJob);
    for (const prefix of ["lab-base", "lab-candidate"]) draw(prefix, currentImage);
    message("Image loaded. No annotations were supplied, so accuracy is not measured.");
  });
  for (const prefix of ["lab-base", "lab-candidate"]) for (const setting of ["size", "confidence", "workers"]) {
    el(prefix + "-" + setting).addEventListener("change", () => {
      if (stream) return;
      ++imageGeneration;
      el("lab-delta").textContent = "Settings changed. Compare again for new measurements.";
      if (currentImage) for (const name of ["lab-base", "lab-candidate"]) draw(name, currentImage);
    });
  }
  window.addEventListener("pagehide", stopCamera);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopCamera(); });
})();
