"""Policies can propose only bounded configurations; they cannot edit the evaluator."""

import json
import os

from .pipeline import Config
from .diagnostics import paired_diagnostics

SYSTEM = """You are the experiment planner for VisionProof, an OpenCV PCB-defect-localization workbench.
Inputs are approximately aligned grayscale test/template pairs. Pipeline: phase-correlation
registration (configurable image size for estimating translation), optional image resize (INTER_AREA),
threshold each, symmetric one-pixel-tolerant difference, morphology open, dilate, area-filtered contours.
Goal: maximize measured throughput while preserving baseline recall/precision within the frozen
requirements. Propose ONE new configuration based on development evidence. A failed candidate
must influence your next experiment. No source changes or evaluator changes are allowed.
The explicit acceptance_floor contains hard constraints. Improving F1 or precision never
compensates for failing recall. Investigate a failed hypothesis rather than repeatedly making
small adjustments without a causal reason. Stage measurements should inform what work to reduce.
Allowed: scale [0.5,0.75,1.0], registration_scale [0.25,0.5,1.0], threshold integer [10,240], min_area integer [1,100],
open_kernel [1,3], workers [1,2,4], threads [1,2,4]. If workers>1, threads must be1.
Use the propose_experiment tool. Pick a base_observation index and change exactly ONE parameter.
The baseline is observation 0. The best ACCEPTED observation is the safest parent for a speed experiment.
Do not inherit a failed configuration when testing a change that cannot plausibly repair its failed metric.
Image scale changes defect pixels; registration_scale changes only translation estimation resolution.
Workers and threads affect throughput, not the detector's semantics. They cannot fix recall loss.
Use paired newly-missed counts and size buckets to understand lost detections. Use stage shares to
identify expensive work. Return a hypothesis grounded in measured facts, and a falsification condition.
Do not claim an outcome before running an experiment. Do not reuse a measured config.
"""

EXPERIMENT_SCHEMA = {"type": "object", "properties": {
    "base_observation": {"type": "integer", "minimum": 0},
    "parameter": {"type": "string", "enum": list(Config().to_dict())},
    "value": {"type": "number"}, "hypothesis": {"type": "string"},
    "falsification": {"type": "string"}},
    "required": ["base_observation", "parameter", "value", "hypothesis", "falsification"],
    "additionalProperties": False}


class ProposalError(ValueError):
    """Keep protocol evidence private while exposing a concise validation reason."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def materialize(proposed: dict, history: list[dict], config_type=Config,
                float_parameters=('scale', 'registration_scale')):
    if set(proposed) != set(EXPERIMENT_SCHEMA["required"]):
        raise ValueError("proposal must use the declared experiment schema")
    parent = proposed["base_observation"]
    if type(parent) is not int or not 0 <= parent < len(history):
        raise ValueError("invalid base observation")
    parameter, value = proposed["parameter"], proposed["value"]
    if parameter not in config_type().to_dict() or type(value) not in (float, int):
        raise ValueError("invalid experiment parameter or value")
    if parameter not in float_parameters:
        if int(value) != value:
            raise ValueError("parameter requires an integer")
        value = int(value)
    config = config_type(**{**history[parent]["result"]["config"], parameter: value})
    if any(config.to_dict() == item["result"]["config"] for item in history):
        raise ValueError("model proposed an already measured configuration")
    if not str(proposed["hypothesis"]).strip() or not str(proposed["falsification"]).strip():
        raise ValueError("hypothesis and falsification must be nonempty")
    return config


def summary(result: dict) -> dict:
    return {key: result[key] for key in ("config", "quality", "performance", "stages_ms", "per_class")}


def choose(provider: str, history: list[dict], requirements: dict, model: str | None = None, pipeline=None) -> dict:
    config_type = pipeline.config_type if pipeline else Config
    float_parameters = pipeline.float_parameters if pipeline else ('scale', 'registration_scale')
    schema = {**EXPERIMENT_SCHEMA, 'properties': {**EXPERIMENT_SCHEMA['properties'],
              'parameter': {'type': 'string', 'enum': list(config_type().to_dict())}}}
    system = SYSTEM
    if pipeline and pipeline.kind == 'pedestrian-yolox':
        system = ('You plan bounded OpenCV experiments. ' + pipeline.instruction + '\n'
                  'Use development evidence only. Propose exactly one new parameter change to a '
                  'measured parent using propose_experiment. Preserve the hard recall, precision and '
                  'p95 floors. Speed or better precision cannot compensate for lost recall. '
                  'Use paired diagnostics and measured stage costs; respond to failed experiments. '
                  'Provide a falsifiable hypothesis, never claim an unmeasured result. '
                  'Do not reuse measured configurations or modify source or requirements.')
    context = {"requirements": requirements, "observations": [
        {**summary(item["result"]), "gate": item["gate"]} for item in history]}
    baseline = history[0]["result"]
    context["acceptance_floor"] = {
        "minimum_recall": baseline["quality"].get("recall", 0) - requirements.get("max_recall_drop", 0),
        "minimum_precision": baseline["quality"].get("precision", 0) - requirements.get("max_precision_drop", 0),
        "maximum_p95_batch_ms": baseline["performance"].get("p95_batch_ms", 0) * requirements.get("max_p95_ratio", 1),
    }
    context["paired_diagnostics"] = [
        {"observation": index, **paired_diagnostics(baseline, item["result"])}
        for index, item in enumerate(history) if "examples" in baseline and "examples" in item["result"]]
    context["best_accepted_observation"] = min(
        (i for i, item in enumerate(history) if item["gate"]["passed"]),
        key=lambda i: history[i]["result"]["performance"].get("median_batch_ms", float("inf")))
    if provider == "rules":
        if pipeline and pipeline.kind == 'pedestrian-yolox':
            for config in pipeline.candidates:
                if not any(config.to_dict() == item['result']['config'] for item in history):
                    rejected = any(not item['gate']['passed'] for item in history)
                    return {'config': config.to_dict(), 'provider': 'rules', 'model': None,
                            'hypothesis': ('Prior evidence rejected a candidate; preserve detector semantics '
                                           'when testing image-level concurrency.' if rejected and config.workers > 1 else
                                           'Test this predeclared DNN configuration; the independent quality gate '
                                           'will detect people lost by reducing inference work.')}
            raise ValueError('DNN control exhausted its predeclared candidates')
        # Explicit non-LLM control: branch on actual rejection; never pass it off as model reasoning.
        tried = [item["result"]["config"] for item in history]
        rejected_scale = any(not item["gate"]["passed"] and item["result"]["config"]["scale"] < 1 for item in history)
        registration_dominates = history[0]["result"]["stages_ms"].get("registration", 0) > sum(
            v for k, v in history[0]["result"]["stages_ms"].items() if k != "registration")
        possibilities = ([Config(registration_scale=0.5), Config(registration_scale=0.25),
                          Config(registration_scale=0.5, workers=2), Config(workers=2), Config(workers=4)]
                         if rejected_scale and registration_dominates else
                         [Config(workers=2), Config(workers=4), Config(threads=2), Config(threads=4)]
                         if rejected_scale else [Config(scale=0.5), Config(scale=0.75), Config(workers=2), Config(workers=4)])
        for config in possibilities:
            if config.to_dict() not in tried:
                return {"config": config.to_dict(), "hypothesis":
                        ("Reduced-scale candidate failed. Registration dominates measured time; estimate its translation at lower resolution while preserving full-resolution defect localization."
                         if rejected_scale and registration_dominates else
                         "A reduced-scale candidate failed the gate. Preserve pixels and test image-level concurrency."
                         if rejected_scale else "Test reduced work per image; the frozen gate will check small-defect loss."),
                        "provider": "rules", "model": None}
        raise ValueError("rule controller exhausted its candidate set")
    payload = json.dumps(context)
    if provider == "bedrock":
        import boto3
        if not model:
            raise ValueError("--model must name an accessible Bedrock model or inference profile")
        response = boto3.client("bedrock-runtime").converse(
            modelId=model, system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": payload}]}],
            inferenceConfig={"maxTokens": 1800, "temperature": 0},
            toolConfig={"tools": [{"toolSpec": {"name": "propose_experiment",
                "description": "Propose one bounded parameter change for independent evaluation.",
                "inputSchema": {"json": schema}}}], "toolChoice": {"auto": {}}})
        content = response["output"]["message"]["content"]
        calls = [block["toolUse"] for block in content if "toolUse" in block]
        text = json.dumps(content)
        if len(calls) != 1 or calls[0]["name"] != "propose_experiment":
            raise ValueError("model must call propose_experiment exactly once")
        proposed = calls[0]["input"]
        usage = response.get("usage")
    elif provider == "anthropic":
        from anthropic import Anthropic
        if not model:
            raise ValueError("--model must name an accessible Anthropic model")
        response = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
            model=model, max_tokens=1800, temperature=0, system=system,
            tools=[{"name": "propose_experiment", "description": "Propose one bounded parameter change.",
                    "input_schema": schema}],
            tool_choice={"type": "tool", "name": "propose_experiment"},
            messages=[{"role": "user", "content": payload}])
        calls = [block for block in response.content if block.type == "tool_use"]
        text = json.dumps([block.model_dump() for block in response.content])
        if len(calls) != 1:
            raise ValueError("model must propose exactly one experiment")
        proposed = calls[0].input
        usage = response.usage.model_dump()
    else:
        raise ValueError(f"unknown provider {provider}")
    try:
        config = materialize(proposed, history, config_type, float_parameters)
    except ValueError as error:
        raise ProposalError(str(error), {'proposal': proposed, 'raw_response': text, 'usage': usage,
                                        'system_prompt': system, 'request_context': context}) from error
    return {"config": config.to_dict(), "hypothesis": str(proposed["hypothesis"]),
            "provider": provider, "model": model, "usage": usage, "raw_response": text,
            "system_prompt": system, "request_context": context, "experiment": proposed}
