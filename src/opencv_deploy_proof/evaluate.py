"""Frozen, class-agnostic localization scorer and repeatable batch measurements."""

import json
import platform
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .data import canonical_digest, digest, load_manifest
from .pipeline import Config, detect


def iou(a, b) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def match(predictions, labels, threshold: float = 0.33) -> dict:
    pairs = sorted(((iou(p, truth[:4]), i, j) for i, p in enumerate(predictions)
                    for j, truth in enumerate(labels)), reverse=True)
    used_predictions, used_labels = set(), set()
    for score, i, j in pairs:
        if score > threshold and i not in used_predictions and j not in used_labels:
            used_predictions.add(i)
            used_labels.add(j)
    return {"tp": len(used_labels), "fp": len(predictions) - len(used_predictions),
            "fn": len(labels) - len(used_labels),
            "missed_indices": sorted(set(range(len(labels))) - used_labels)}


def metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def fingerprint() -> dict:
    modules = ["data.py", "pipeline.py", "evaluate.py"]
    return {"opencv": cv2.__version__, "opencv_module": cv2.__file__,
            "opencv_build": cv2.getBuildInformation(), "numpy": np.__version__,
            "python": sys.version, "system": platform.platform(), "machine": platform.machine(),
            "code_sha256": {name: digest(Path(__file__).parent / name) for name in modules}}


class Evaluator:
    """Load and hash-check inputs before experiments; the policy receives dev summaries only."""

    def __init__(self, manifest: Path, split: str, repeats: int = 9):
        if split not in ("calibration", "development", "test") or repeats < 3:
            raise ValueError("invalid split or repeats < 3")
        self.manifest = load_manifest(manifest)
        self.split, self.repeats = split, repeats
        self.rows = []
        for row in self.manifest["records"]:
            if row["split"] == split:
                image = cv2.imread(str(manifest.parent / row["image"]), cv2.IMREAD_GRAYSCALE)
                template = cv2.imread(str(manifest.parent / row["template"]), cv2.IMREAD_GRAYSCALE)
                labels = [[int(v) for v in line.split()] for line in
                          (manifest.parent / row["label"]).read_text().splitlines() if line.strip()]
                if image is None or template is None or any(len(label) != 5 for label in labels):
                    raise ValueError(f"invalid example {row['id']}")
                self.rows.append((row, image, template, labels))
        if not self.rows:
            raise ValueError("empty evaluation split")

    def run(self, config: Config) -> dict:
        cv2.setNumThreads(config.threads)

        def execute(row):
            return detect(row[1], row[2], config)

        durations, outputs = [], []
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            for iteration in range(self.repeats + 2):
                start = perf_counter()
                outputs = list(pool.map(execute, self.rows)) if config.workers > 1 else [execute(r) for r in self.rows]
                duration = (perf_counter() - start) * 1000
                if iteration >= 2:
                    durations.append(duration)
        totals = {"tp": 0, "fp": 0, "fn": 0}
        examples, per_class = [], {}
        for (row, image, _, labels), output in zip(self.rows, outputs):
            scored = match(output["boxes"], labels)
            for key in totals:
                totals[key] += scored[key]
            for i, label in enumerate(labels):
                item = per_class.setdefault(str(label[4]), {"total": 0, "missed": 0})
                item["total"] += 1
                item["missed"] += int(i in scored["missed_indices"])
            examples.append({"id": row["id"], "image": row["image"], "shape": list(image.shape),
                             "labels": labels, "predictions": output["boxes"], **scored})
        median = statistics.median(durations)
        result = {"config": config.to_dict(), "split": self.split,
                  "manifest_sha256": self.manifest["sha256"], "quality": metrics(**totals),
                  "per_class": per_class, "examples": examples,
                  "performance": {"batch_ms": durations, "median_batch_ms": median,
                                  "p95_batch_ms": float(np.percentile(durations, 95)),
                                  "images_per_second": len(self.rows) * 1000 / median,
                                  "image_count": len(self.rows), "repeats": self.repeats,
                                  "scope": "warm in-memory batch; excludes decode, networking, cold start and service queueing"},
                  "stages_ms": {key: statistics.median(o["stages_ms"][key] for o in outputs)
                                for key in outputs[0]["stages_ms"]},
                  "environment": fingerprint()}
        result["sha256"] = canonical_digest(result)
        return result


@dataclass(frozen=True)
class Requirements:
    max_recall_drop: float = 0.02
    max_precision_drop: float = 0.02
    max_p95_ratio: float = 1.25


def gate(candidate: dict, baseline: dict, requirements: Requirements) -> dict:
    if candidate["split"] != baseline["split"] or candidate["manifest_sha256"] != baseline["manifest_sha256"]:
        raise ValueError("cannot compare different splits or datasets")
    checks = {
        "recall": candidate["quality"]["recall"] >= baseline["quality"]["recall"] - requirements.max_recall_drop,
        "precision": candidate["quality"]["precision"] >= baseline["quality"]["precision"] - requirements.max_precision_drop,
        "p95_batch": candidate["performance"]["p95_batch_ms"] <= baseline["performance"]["p95_batch_ms"] * requirements.max_p95_ratio,
    }
    return {"passed": all(checks.values()), "checks": checks, "requirements": asdict(requirements),
            "scope": "empirical relative-to-baseline benchmark gate, not a safety guarantee"}


def write_result(path: Path, result: dict):
    path.write_text(json.dumps(result, indent=2))
