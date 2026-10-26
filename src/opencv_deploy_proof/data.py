"""Fetch a pinned public benchmark and split by source board group."""

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

COMMIT = "08e98c4db5922613fb97176eb3d6497d48260cb1"
SOURCE = "https://github.com/tangsanli5201/DeepPCB"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(root: Path, limit: int = 80) -> Path:
    if limit < 1:
        raise ValueError("limit must be positive")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "source.json"
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix="deep-pcb-") as temp:
            archive = Path(temp) / "source.tar.gz"
            request = urllib.request.Request(
                f"https://codeload.github.com/tangsanli5201/DeepPCB/tar.gz/{COMMIT}",
                headers={"User-Agent": "opencv-deploy-proof/0.1"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            archive_hash = digest(archive)
            with tarfile.open(archive) as tar:
                for member in tar:
                    relative = Path(*Path(member.name).parts[1:])
                    if not member.isfile() or not relative.parts:
                        continue
                    if relative.parts[0] not in ("PCBData", "LICENSE", "README.md"):
                        continue
                    target = (root / relative).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError("unsafe archive path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            marker.write_text(json.dumps({"source": SOURCE, "commit": COMMIT,
                                         "archive_sha256": archive_hash}, indent=2))
    if json.loads(marker.read_text())["commit"] != COMMIT:
        raise ValueError("dataset source marker is not the pinned commit")
    groups = sorted((root / "PCBData").glob("group*"), key=lambda p: canonical_digest(p.name))
    if len(groups) < 5:
        raise ValueError("incomplete dataset: need at least five source groups")
    group_splits = {"test": groups[:2], "development": groups[2:4], "calibration": groups[4:]}
    records = []
    for split, selected in group_splits.items():
        per_group = []
        for group in selected:
            rows = []
            for image in group.glob("*/*_test.jpg"):
                stem = image.name.removesuffix("_test.jpg")
                template = image.with_name(stem + "_temp.jpg")
                label = group / (image.parent.name + "_not") / (stem + ".txt")
                if not template.is_file() or not label.is_file():
                    raise ValueError(f"missing pair/label for {stem}")
                paths = {"image": image, "template": template, "label": label}
                rows.append({"id": stem, "group": group.name, "split": split,
                             **{k: str(p.relative_to(root)) for k, p in paths.items()},
                             "sha256": {k: digest(p) for k, p in paths.items()}})
            per_group.append(sorted(rows, key=lambda r: canonical_digest(r["id"])))
        # Round-robin sampling avoids allowing one large source board to dominate a split.
        combined = [rows[i] for i in range(max(map(len, per_group)))
                    for rows in per_group if i < len(rows)]
        records.extend(combined[:limit])
    manifest = {"source": SOURCE, "commit": COMMIT,
                "split_method": "sha256-ordered source-board groups; custom split, not official leaderboard split",
                "group_splits": {k: [p.name for p in v] for k, v in group_splits.items()},
                "limitations": ["Some defects were manually introduced by the dataset authors.",
                                "Evidence applies to this benchmark, not real factory deployment."],
                "records": records}
    manifest["sha256"] = canonical_digest(manifest)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop("sha256")
    if canonical_digest(value) != expected:
        raise ValueError("manifest digest mismatch")
    value["sha256"] = expected
    memberships = {}
    for row in value["records"]:
        previous = memberships.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("source-board leakage across splits")
        for key in ("image", "template", "label"):
            file = (path.parent / row[key]).resolve()
            if not file.is_relative_to(path.parent.resolve()) or digest(file) != row["sha256"][key]:
                raise ValueError(f"dataset file verification failed: {row['id']} {key}")
    return value
