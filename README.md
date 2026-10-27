# VisionProof

**Faster can mean blind.** Inspect the detections an optimization loses, then block changes
that violate the quality contract. Solo project by Laksh Goyal.

[Public measured replay](https://visionproof.vercel.app/) ·
[Architecture and 2:55 demo video](https://visionproof.vercel.app/about.html) ·
[GitHub Actions](https://github.com/lgoyal6/visionproof/actions)

The public endpoint is a read-only replay with illustrative image samples. It does not run
inference or qualification jobs. The local app below runs both. Use **Guided demo** to inspect
the recorded change that lost 559 baseline detections, then the passing change that retained them.

## Architecture

![VisionProof architecture: browser and Python server run both OpenCV 5 configurations locally; an independent evaluator and gate produce the evidence report; Amazon Bedrock proposes bounded settings; recorded EC2 x86, Graviton and COOL benchmarks; GitHub Actions runs the same check on pull requests](docs/architecture.png)

Live measurements run locally with OpenCV 5. Amazon Bedrock only proposes bounded settings and cannot edit the evaluator. AWS EC2 x86, Graviton and COOL results are recorded evidence from short-lived workers. GitHub Actions runs `vision-proof check` on pull requests.

An OpenCV qualification workbench for developers who need faster image processing without
silently losing detections. An experiment planner proposes bounded changes to a PCB inspection
pipeline. An independent evaluator checks recall, precision and warm batch latency before a
candidate becomes eligible for preview.

The workbench includes side-by-side detections, missed-defect overlays and crops, real local
inference, experiment hypotheses and rejection reasons, a quality/throughput plot, bounded
qualification jobs, recorded AWS comparisons, and a downloadable reproduction bundle.

It supports two explicit adapters: classical PCB image-pair inspection and OpenCV Zoo YOLOX
pedestrian localization on single images. Both share the scorer, recall/precision/latency
contract, selection protocol, evidence reader, workbench and CI gate. It does not accept arbitrary
repositories or execute uploaded Python. The planner has no evaluator-editing tool. This is
logical separation, not a hostile-code sandbox.

## Start locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen --extra test
uv run python -m pytest -q
./scripts/demo.sh
```

The script downloads a pinned public dataset into `~/.codex/work/visionproof`, declares the
split, runs a deterministic qualification, then serves the workbench at
[127.0.0.1:8765](http://127.0.0.1:8765/). It never subscribes to Marketplace or creates AWS resources.
The default controller is labeled as deterministic, not an LLM. A successful qualification is
required to start the preview. Repeated runs on the same dataset are repeated validation, not
new holdout evidence.

For the real Amazon Bedrock planner, use your existing AWS profile and an accessible model:

```sh
AWS_DEFAULT_REGION=us-east-1 \
VISIONPROOF_PROVIDER=bedrock \
VISIONPROOF_MODEL=us.amazon.nova-2-lite-v1:0 ./scripts/demo.sh
```

This makes metered Bedrock calls. There is no silent fallback on API or protocol failure.
The UI enables Bedrock jobs only when the server starts with a configured model. Opening a
page or viewing historical evidence never invokes a model.

## Reproduce individual stages

### Live comparison, imported data and webcam

In the **local app**, open **Live comparison**. Choose baseline and candidate input size,
confidence and batch-worker settings. Compare one labeled image, upload an unlabeled PNG/JPEG,
or start a webcam. Both configurations receive the same pixels. The model processes grayscale
images; uploads and camera frames are normalized to a maximum 1280-pixel edge in the browser.
Image timing is a warm single-image preview. Worker counts affect dataset batches, not a
single-frame speed claim. The historical qualification badge is hidden in this view.

This view requires an existing qualified `pedestrian-yolox` run under the server's evidence
root. The pedestrian setup below downloads and pins the model. The original PCB adapter stays
available in the existing inspection and qualification views.

Use **Import dataset** for a ZIP containing `annotations.json` at the archive root and the image
files it names. The supported COCO subset has one category named `person`, unique integer image
and annotation IDs, exact image dimensions, and absolute pixel boxes `[x, y, width, height]`.
Other declared categories are ignored. Crowd boxes are rejected because this scorer does not
implement COCO crowd matching. Example `annotations.json`:

```json
{
  "images": [{"id": 1, "file_name": "images/frame.png", "width": 640, "height": 480}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 80, 50, 150]}]
}
```

Imports accept 1-64 images, at least one person annotation, ZIPs up to 32 MiB, at most 64 MiB
expanded, and at most 4 megapixels per image / 32 megapixels total. Unsafe paths, duplicate
entries, invalid boxes and mismatched dimensions are rejected. Original ZIPs, normalized images,
labels, model and hash-bound manifests stay under the private evidence root's `imports/` folder.
Imported annotations are user assertions, not independently verified ground truth.

**Evaluate dataset** freezes the two configurations and your recall, precision and latency
limits before three randomized blocks of three timed batches. It uses the existing scorer
(one-to-one matching at IoU > 0.33) and exports the frozen contract, raw blocks, current report
and original inputs' manifest. Imported checks use a single test set with no search or new
holdout claim. These private checks are excluded from public replay exports.

**Webcam** uses the browser's camera picker, including a connected Logitech camera. It samples
one frame at a time, runs both configurations, then requests the next frame. This is interactive
comparison, not a promised video frame rate. No audio is requested and camera frames are not
saved. Stop, changing source, leaving the tab, hiding the page and navigating away release the
camera tracks. Camera permission denial is reported in the UI. Without annotations, uploads
and webcam frames show prediction disagreement and processing time, with no accuracy verdict.

### CI regression gate

```sh
uv run vision-proof check --manifest /path/to/data/manifest.json \
  --baseline-config configs/base.json --candidate-config configs/pcb.json \
  --rounds 5 --repeats 5 --seed 1729 --out /path/to/new-ci-report
```

The baseline argument is optional and defaults to the adapter's full-resolution single-worker
configuration. Exit **0** means the measured candidate passed; **2** means a measured regression;
**3** means invalid input or execution failure. Failed checks still produce `summary.md`, a
signed JSON summary, frozen contract, raw blocks, paired lost detections and a verified journal.
An execution error produces no passing qualification. Use a new output directory per check.

`.github/workflows/visionproof.yml` runs both pipelines on pull requests, reads the baseline
configuration from the PR base revision, adds the summary to the job, and uploads evidence
even if qualification fails. No AWS credentials, subscriptions or paid APIs are needed.
The workflow qualifies **bounded configuration changes against the current implementation**.
It does not compare arbitrary source revisions or configure GitHub branch protection. A missing
baseline config on first adoption uses the adapter default. No remote CI run is claimed until
the repository is published and Actions executes it.

### Second pipeline: pedestrian localization

```sh
uv run vision-proof prepare-pedestrians --data /path/to/pedestrian-data --limit 48
uv run vision-proof proof --manifest /path/to/pedestrian-data/manifest.json \
  --out /path/to/new-pedestrian-run --provider rules --budget 4 --repeats 5
uv run vision-proof serve --manifest /path/to/pedestrian-data/manifest.json \
  --run /path/to/new-pedestrian-run --evidence-root /path/to
```

The adapter fetches the original Penn-Fudan dataset and the Apache-2.0 YOLOX model from
OpenCV Zoo. Both downloads are pinned by SHA256. Model inputs use grayscale replicated to BGR,
top-left letterboxing and the upstream grid/stride decoder; only the COCO person class is scored.
This is a distinct neural OpenCV DNN/CPU workload. The model is pretrained, not trained here.
Boxes are derived from public instance masks. Development uses Fudan images and test uses Penn
images, sampled before measuring outcomes; source-location separation is not a guarantee of
person-identity separation or training-data independence. See [NOTICE.md](NOTICE.md).

The same workbench can catalog both adapters when their run folders are below the evidence root.
Selecting a pedestrian run enables real inference on that detector. Starting a qualification uses
the active run's registered dataset. Templates are disabled for this single-image adapter.

### Randomized replication on reserved PCB boards

```sh
uv run vision-proof prepare-reserved \
  --original /path/to/data/manifest.json --original /path/to/data/manifest-fresh.json \
  --out /path/to/data/manifest-reserved.json --limit 160
uv run vision-proof compare --manifest /path/to/data/manifest-reserved.json \
  --configs /path/to/frozen-configs.json --rounds 7 --repeats 7 --seed 1729 \
  --out /path/to/new-replication
```

`frozen-configs.json` maps `baseline` and named candidates to full adapter configurations. Declare
them before measuring the reserved test set. Prior development/test board groups and **all**
previously registered image IDs are excluded. Remaining boards appeared only in the prior
calibration pool. They had earlier baseline calibration measurements. These are fresh images on calibration-exposed boards, not unseen boards or a new external dataset.

Each block evaluates every configuration. Shuffled rotations balance measurement positions;
the complete schedule, configurations, source-board memberships and dataset digest are saved
before the evaluator opens test images. Reports preserve every raw batch time and compute a
95% percentile bootstrap interval for median **paired block speedup** using 10,000 resamples.
The gate checks quality and the median of per-block p95 batch times. That statistic is not
request p95. Per-group quality and newly lost/recovered annotations accompany pooled metrics.
Intervals describe local timing variability; they do not certify independent-machine or unseen
dataset generalization. Only three reserved DeepPCB groups remain, with unequal sample counts.
Repeating fixed configurations is a replication, not a repeated optimizer-search comparison.

The **CI & replication** workbench tab verifies the raw evidence, shows uncertainty, explains
blocked changes, previews lost detections and exports the study. Empty or incomplete studies
never appear as successful qualifications.

### Original qualification commands

```sh
uv run vision-proof prepare --data "$HOME/.codex/work/visionproof/data" --limit 80
uv run vision-proof prepare-fresh \
  --original "$HOME/.codex/work/visionproof/data/manifest.json" \
  --out "$HOME/.codex/work/visionproof/data/manifest-fresh.json" --limit 64
uv run vision-proof proof \
  --manifest "$HOME/.codex/work/visionproof/data/manifest-fresh.json" \
  --out "$HOME/.codex/work/visionproof/new-run" --provider rules --budget 4 --repeats 9
uv run vision-proof serve \
  --manifest "$HOME/.codex/work/visionproof/data/manifest-fresh.json" \
  --run "$HOME/.codex/work/visionproof/new-run" \
  --evidence-root "$HOME/.codex/work/visionproof"
```

Use fresh output directories. `prepare-fresh` refuses to overwrite a declared manifest. Run
`uv run vision-proof --help` for all commands. The workbench's **New qualification** button runs
one bounded job at a time, persists progress, and opens its actual result. Live inference pauses
while qualification measures performance to avoid local timing contention.

## How it works

1. Verify source pin, file hashes and source-board group separation.
2. Register grayscale test/template pairs using phase correlation.
3. Compare thresholded pixels with one-pixel edge tolerance, morphology and contour localization.
4. Give the planner development stage timings, hard acceptance floors, and paired lost/recovered
   annotation counts. Bedrock tool use proposes one parameter change to a measured parent.
5. Evaluate the proposal independently. Preserve rejected configurations and raw model decisions.
6. Freeze the fastest accepted development candidate, compare with an equal-budget fixed sweep,
   then evaluate the selected configurations on test images.
7. Verify source/runtime/configuration bindings before allowing the local inference preview.
8. Export source, configuration, manifest, measured outputs and a qualification report.

The detailed implementation and verification plan is in [DESIGN.md](DESIGN.md).

## What the measured runs showed

These are observations from the preserved October 26, 2026 experiments, not performance promises.
The private artifacts live outside the repository; a fresh clone produces its own measurements.

| Experiment | Observed result |
| --- | --- |
| Original real LLM searches | All 12 proposals across Nova Pro and Nova 2 failed recall; baseline retained. |
| Diagnostic Bedrock search, 64 fresh test images | Two resized candidates rejected. Full-resolution selected configuration passed, 3.73x local baseline throughput. |
| Same-budget fixed sweep on that run | About 1.2% faster than the selected agent configuration, with better precision. No LLM superiority established. |
| Reserved-board replication, 166 fresh images, seven randomized blocks | Frozen Bedrock selection passed: 3.34x paired baseline speedup, 95% local timing interval 2.11-4.10x. |
| Frozen Bedrock selection versus fixed sweep on reserved boards | 1.07x median paired speedup, interval 0.81-1.67x includes parity. Sweep retained better recall and precision. No LLM search advantage established. |
| Second pipeline, YOLOX on 48 Penn test images | Selected four-worker configuration preserved baseline TP138/FP41/FN1 and passed. Initial sequential measurement: 3.16x baseline. Deterministic controller, not an LLM result. |
| YOLOX replication, five randomized blocks | Selected: 3.13x paired baseline speedup, interval 2.53-3.40x, identical detections. Smaller input: 3.33x but lost ten baseline detections and failed recall despite 94.8% precision. |
| Local CI commands on calibration-exposed PCB boards | Faster reduced-resolution change exited 2 and retained its failure report. Four-worker change exited 0. See GitHub Actions for separately measured remote results. |
| Earlier AWS c8g.large, pip OpenCV 5.0.0 | Selected configuration: 191.4 ms per 80-image warm batch. |
| Earlier AWS c7i.large, pip OpenCV 5.0.0 | Same configuration: 201.6 ms per batch. |
| AWS c8g.large, COOL OpenCV 5.1.0-dev | Same earlier configuration: 183.7 ms per batch; identical scored detections to pip. |

The later Bedrock-selected configuration was tested locally, not on AWS. The cloud comparison
uses the earlier rules-selected configuration. COOL and pip also used different NumPy versions
and base images. The roughly 4% cloud latency difference does not isolate COOL's contribution.
Workers were terminated and the COOL trial was cancelled after measurement.

The reserved-board replication has 2 images from group12000, 68 from group50600 and 96 from
group77000. Those boards had earlier baseline calibration measurements; every prior registered
image is excluded. The new images are fresh, but source boards are not unseen. These sample sizes and only three source groups limit distribution claims.
The new pedestrian Bedrock attempt accepted its first proposal but stopped when the second
proposal reused an already measured configuration. That incomplete run is not qualified. The
working second-pipeline qualification above uses the explicitly labeled deterministic controller.

All latency measurements are warm, in-memory batches. They exclude decoding, network transit,
startup, idle time and service queues. Compute-only price estimates are not invoices or all-in
cost savings. Marketplace software fees, storage, networking and model calls are separate.

## Dataset and quality boundaries

[DeepPCB](https://github.com/tangsanli5201/DeepPCB/tree/08e98c4db5922613fb97176eb3d6497d48260cb1)
is pinned to commit `08e98c4db5922613fb97176eb3d6497d48260cb1`, with its MIT license. The application
locates defects without classifying the six defect types. Some dataset defects were manually
introduced by the authors; this benchmark is not factory validation.

The initial custom protocol used 80 pairs each for calibration, development and test. The v2
protocol excludes all 240 original image IDs and samples 64 development and 64 test images from
disjoint groups in the previous calibration pool. Images are new, but source boards are not
wholly unseen. It is not the official leaderboard protocol. Selection sees development summaries;
test evaluation starts only after the selection artifact is written.

The fixed gate permits at most 0.02 absolute recall loss, 0.02 precision loss and a 1.25x p95
batch ratio against baseline. Matching is class-agnostic, one-to-one, with IoU strictly above
0.33. Baseline quality itself is imperfect. Passing a relative gate is not a safety guarantee.
A single sequential search is sensitive to timing/order and does not establish optimizer superiority.

## Evidence and integrity

Each run saves a contract, baseline, proposals, candidate measurements, diagnostics, frozen
selection, sweep measurements, test results and verdict. New runs also save a hash-linked event
journal. The model's raw tool response, prompt, context and token usage remain in the private run.
The browser receives a compact trace, not credentials.

The reader checks artifact digests, candidate membership, configuration/contract/dataset bindings
and recomputed gates. Hashes detect accidental changes; they are not signatures against an attacker
who can rewrite every artifact. The server is loopback-only, allows no arbitrary file paths,
checks Host and Origin for mutation, bounds request size and job budgets, and renders model text
as text rather than HTML. Inference rejects runtime/source drift and failed qualification.

## Cloud measurement

Install cloud extras and use an authenticated profile:

```sh
uv sync --frozen --extra cloud --extra test
uv run python scripts/cloud_run.py --help
uv run python scripts/compare_cloud.py --help
```

`cloud_run.py` supports pip on x86, pip on Graviton and an already entitled COOL AMI. It never
subscribes automatically. Use `--plan-only` to inspect the plan. COOL loads the vendor environment
under `/opt/cool`, refuses an unproven runtime, and does not replace the optimized OpenCV package.
A temporary worker is restricted to SSH from the client IPv4, has a shutdown timer and uses IMDSv2.
The runner records resources, collects results, requests termination and removes temporary keys
and security groups. Fresh AWS readback is required to verify termination and EBS cleanup.

With a registered `aws-cool-comparison.json` and its original worker result folders under the
configured evidence root, the AWS tab verifies and displays those measurements. Otherwise it
shows an explicit empty state. A `cool-cleanup-verification.json` record supplies the recorded
cleanup status, separately from successful evaluation. The tab is not a live AWS monitor.

## Container

```sh
docker build -t visionproof:local .
docker run --rm visionproof:local --help
```

The image runs as an unprivileged user. Mount datasets read-only and a separate writable output
directory for evaluation. Local Linux Arm evaluation verifies compatibility, not AWS performance.
Cloud/model extras are optional and are not installed in the default container.

## Tests and demo

`uv run pytest -q` covers matching, alignment, quality gates, input integrity, paired diagnostics,
structured proposals, event tampering, selection binding, HTTP boundaries, exported evidence,
job exclusion and SSH readiness recovery. Cloud tests skip when optional SDK dependencies are absent.

The **Guided demo** button plays recorded decisions with explicit labeling. Real inference and
new local qualification remain separate actions. Historical model failures are retained in the
run navigator. A finished screen recording is generated from the working app during delivery;
it is not a fabricated vision result or public deployment.
