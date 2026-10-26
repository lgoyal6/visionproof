"""Run bounded development experiments, freeze a choice, then open the test split."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .data import canonical_digest
from .evaluate import Evaluator, Requirements, gate, write_result
from .pipeline import Config
from .policy import choose
from .diagnostics import paired_diagnostics
from .journal import append_event
from .adapters import pipeline_for


def proof(manifest: Path, out: Path, provider="rules", model=None, budget=4, repeats=9) -> dict:
    pipeline = pipeline_for(manifest)
    evaluator, config_type = pipeline.evaluator, pipeline.config_type
    if budget < 1 or budget > 12:
        raise ValueError("experiment budget must be between 1 and 12")
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be a new/empty directory so evidence cannot be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    requirements = Requirements()
    contract = {"created_utc": datetime.now(timezone.utc).isoformat(), "requirements": asdict(requirements),
                "metric": "class-agnostic one-to-one localization, IoU > 0.33",
                "pipeline": pipeline.kind,
                "experiment_budget": budget, "provider": provider, "model": model,
                "manifest_path": str(manifest.resolve()),
                "cost_status": "unmeasured; local timing is not AWS cost evidence"}
    (out / "contract.json").write_text(json.dumps(contract, indent=2))
    journal = out / "events.jsonl"
    append_event(journal, "started", {"provider": provider, "budget": budget})
    development = evaluator(manifest, "development", repeats)
    baseline = development.run(config_type())
    baseline_gate = {"passed": True, "checks": {}, "requirements": asdict(requirements)}
    history = [{"decision": {"hypothesis": "unchanged baseline"}, "result": baseline, "gate": baseline_gate}]
    write_result(out / "baseline-development.json", baseline)
    append_event(journal, "baseline", {"quality": baseline["quality"], "batch_ms": baseline["performance"]["median_batch_ms"]})
    for index in range(budget):
        append_event(journal, "planning", {"experiment": index + 1})
        try:
            decision = choose(provider, history, asdict(requirements), model, pipeline=pipeline)
        except ValueError as error:
            details = {'experiment': index + 1, 'error': str(error),
                       'evidence': getattr(error, 'evidence', {})}
            write_result(out / f'planner-error-{index + 1:02d}.json', details)
            append_event(journal, 'failed', {'experiment': index + 1, 'error': str(error)})
            raise
        append_event(journal, "proposed", {"experiment": index + 1, "config": decision["config"], "hypothesis": decision["hypothesis"]})
        measured = development.run(config_type(**decision["config"]))
        entry = {"decision": decision, "result": measured, "gate": gate(measured, baseline, requirements),
                 "diagnostics": paired_diagnostics(baseline, measured)}
        history.append(entry)
        write_result(out / f"agent-{index + 1:02d}.json", entry)
        append_event(journal, "measured", {"experiment": index + 1, "gate": entry["gate"],
                     "quality": measured["quality"], "batch_ms": measured["performance"]["median_batch_ms"]})
        print(json.dumps({"experiment": index + 1, "provider": provider, "config": measured["config"],
                          "quality": measured["quality"], "gate": entry["gate"],
                          "batch_ms": measured["performance"]["median_batch_ms"]}), flush=True)
    eligible = [entry for entry in history if entry["gate"]["passed"]]
    best = min(eligible, key=lambda e: e["result"]["performance"]["median_batch_ms"])
    selected = best["result"]["config"]
    # Freeze the candidate BEFORE instantiating the holdout evaluator.
    selection = {"config": selected, "development_result_sha256": best["result"]["sha256"],
                 "contract_sha256": canonical_digest(contract), "manifest_sha256": baseline["manifest_sha256"],
                 "selection_rule": "lowest warm median batch time among development-gate passes"}
    selection["sha256"] = canonical_digest(selection)
    write_result(out / "selection.json", selection)
    append_event(journal, "selection_frozen", {"sha256": selection["sha256"], "config": selected})
    # Same candidate budget, fixed order, no model or outcome-dependent choices.
    fixed = pipeline.candidates
    sweep = []
    for index, config in enumerate(fixed[:budget]):
        append_event(journal, "sweep", {"experiment": index + 1})
        measured = development.run(config)
        entry = {"result": measured, "gate": gate(measured, baseline, requirements)}
        sweep.append(entry)
        write_result(out / f"sweep-{index + 1:02d}.json", entry)
    sweep_eligible = [{"result": baseline, "gate": baseline_gate}] + [x for x in sweep if x["gate"]["passed"]]
    sweep_best = min(sweep_eligible, key=lambda e: e["result"]["performance"]["median_batch_ms"])
    append_event(journal, "holdout_started", {"selection_sha256": selection["sha256"]})
    holdout = evaluator(manifest, "test", repeats)
    test_baseline = holdout.run(config_type())
    test_selected = holdout.run(config_type(**selected))
    test_sweep = holdout.run(config_type(**sweep_best["result"]["config"]))
    for label, result in [("baseline", test_baseline), ("agent", test_selected), ("sweep", test_sweep)]:
        write_result(out / f"{label}-test.json", result)
    final_gate = gate(test_selected, test_baseline, requirements)
    verdict = {
        "provider": provider, "model": model, "pipeline": pipeline.kind, "selection": selection, "holdout_gate": final_gate,
        "baseline_test_quality": test_baseline["quality"], "selected_test_quality": test_selected["quality"],
        "sweep_test_quality": test_sweep["quality"],
        "baseline_test_batch_ms": test_baseline["performance"]["median_batch_ms"],
        "selected_test_batch_ms": test_selected["performance"]["median_batch_ms"],
        "sweep_test_batch_ms": test_sweep["performance"]["median_batch_ms"],
        "speedup_vs_baseline": test_baseline["performance"]["median_batch_ms"] / test_selected["performance"]["median_batch_ms"],
        "speedup_vs_fixed_sweep": test_sweep["performance"]["median_batch_ms"] / test_selected["performance"]["median_batch_ms"],
        "development_candidates": budget, "sweep_candidates": budget,
        "rejected_development_candidates": sum(not e["gate"]["passed"] for e in history[1:]),
        "promotion_status": "preview_eligible" if final_gate["passed"] else "refused_holdout_failed",
        "limitations": [("Baseline is a pretrained neural locator, not production accuracy evidence."
                          if pipeline.kind == 'pedestrian-yolox' else
                          "Baseline is a simple classical locator, not production accuracy evidence."),
                        "Benchmark evidence is finite and does not certify future or industrial accuracy.",
                        "Sequential local runs can have thermal/order effects; speed differences need replication.",
                        "No AWS/COOL result or cost saving is implied by this local run.",
                        "Rules provider is a deterministic diagnostic control, not an LLM agent."],
    }
    verdict["sha256"] = canonical_digest(verdict)
    write_result(out / "verdict.json", verdict)
    append_event(journal, "completed", {"promotion_status": verdict["promotion_status"],
                 "speedup_vs_baseline": verdict["speedup_vs_baseline"], "holdout_gate": final_gate})
    return verdict
