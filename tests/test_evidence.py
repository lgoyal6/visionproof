import copy
import json

import numpy as np
import pytest

from opencv_deploy_proof.data import canonical_digest, digest, load_manifest
from opencv_deploy_proof.evaluate import Requirements, gate, match
from opencv_deploy_proof.pipeline import Config, detect
from opencv_deploy_proof.policy import choose


def observation(config=None, passed=True):
    return {"result": {"config": (config or Config()).to_dict(), "quality": {},
                       "performance": {}, "stages_ms": {}, "per_class": {}},
            "gate": {"passed": passed}}


def test_duplicate_predictions_cannot_inflate_recall():
    scored = match([[0, 0, 10, 10], [0, 0, 10, 10]], [[0, 0, 10, 10, 1]])
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 1, 0)


def test_one_large_box_cannot_match_multiple_defects():
    scored = match([[0, 0, 10, 10]], [[0, 0, 10, 10, 1], [0, 0, 10, 10, 2]])
    assert (scored["tp"], scored["fn"]) == (1, 1)


def test_pipeline_detects_added_shape_and_not_unchanged_pair():
    reference = np.zeros((80, 80), np.uint8)
    image = reference.copy()
    image[20:35, 30:45] = 255
    assert detect(reference, reference, Config())["boxes"] == []
    boxes = detect(image, reference, Config())["boxes"]
    assert len(boxes) == 1
    assert boxes[0][0] <= 30 and boxes[0][2] >= 45


def test_registration_removes_translation_without_hiding_added_defect():
    import cv2
    rng = np.random.default_rng(7)
    reference = np.zeros((160, 160), np.uint8)
    for x, y in rng.integers(10, 130, (20, 2)):
        reference[y:y + 10, x:x + 15] = 255
    moved = cv2.warpAffine(reference, np.float32([[1, 0, 2], [0, 1, -2]]), (160, 160))
    assert detect(moved, reference, Config())["boxes"] == []
    moved[135:150, 135:150] = 255
    answer = detect(moved, reference, Config())
    assert match(answer["boxes"], [[130, 130, 155, 155, 1]])["tp"] == 1


def test_bad_input_pair_is_refused():
    with pytest.raises(ValueError, match="aligned pair"):
        detect(np.zeros((20, 20), np.uint8), np.zeros((10, 10), np.uint8), Config())


def test_preview_refuses_failed_qualification_before_binding_port(tmp_path):
    from opencv_deploy_proof.serve import serve
    manifest = {"records": []}
    manifest["sha256"] = canonical_digest(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verdict = {"promotion_status": "refused_holdout_failed"}
    verdict["sha256"] = canonical_digest(verdict)
    (tmp_path / "verdict.json").write_text(json.dumps(verdict))
    with pytest.raises(ValueError, match="refused"):
        serve(tmp_path / "manifest.json", tmp_path, port=0)


@pytest.mark.parametrize("kwargs", [{"scale": 0.1}, {"workers": 128}, {"threads": 0},
                                    {"workers": 4, "threads": 4}, {"threshold": 255}])
def test_candidate_is_bounded(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_quality_failure_cannot_be_bought_with_speed():
    baseline = {"split": "development", "manifest_sha256": "same", "quality": {"recall": 0.95, "precision": 0.90},
                "performance": {"p95_batch_ms": 100}}
    candidate = copy.deepcopy(baseline)
    candidate["quality"]["recall"] = 0.80
    candidate["performance"]["p95_batch_ms"] = 1
    result = gate(candidate, baseline, Requirements())
    assert not result["passed"] and not result["checks"]["recall"]


def test_gate_refuses_cross_dataset_comparison():
    with pytest.raises(ValueError, match="different splits"):
        gate({"split": "test", "manifest_sha256": "a"},
             {"split": "development", "manifest_sha256": "a"}, Requirements())


def test_rule_control_changes_next_experiment_after_failed_resize():
    initial = [observation()]
    first = choose("rules", initial, {})
    assert first["config"]["scale"] == 0.5
    initial.append(observation(Config(scale=0.5), passed=False))
    second = choose("rules", initial, {})
    assert second["config"]["scale"] == 1 and second["config"]["workers"] > 1


def test_dataset_manifest_checks_file_hashes_and_group_leakage(tmp_path):
    for name in ["image", "template", "label"]:
        (tmp_path / name).write_text("original")
    row = {"id": "one", "group": "board-a", "split": "development",
           **{k: k for k in ["image", "template", "label"]},
           "sha256": {k: digest(tmp_path / k) for k in ["image", "template", "label"]}}
    manifest = {"records": [row]}
    path = tmp_path / "manifest.json"

    def save(value):
        value = copy.deepcopy(value)
        value["sha256"] = canonical_digest(value)
        path.write_text(json.dumps(value))

    save(manifest)
    assert load_manifest(path)["records"][0]["id"] == "one"
    (tmp_path / "image").write_text("changed")
    with pytest.raises(ValueError, match="verification failed"):
        load_manifest(path)
    (tmp_path / "image").write_text("original")
    manifest["records"].append({**row, "id": "two", "split": "test"})
    save(manifest)
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path)
