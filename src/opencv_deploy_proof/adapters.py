"""Two explicit pipeline adapters sharing a localization evidence contract."""

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluate import Evaluator, fingerprint
from .pipeline import Config, detect


@dataclass(frozen=True)
class Pipeline:
    kind: str
    title: str
    config_type: type
    evaluator: type
    detect: object
    fingerprint: object
    candidates: tuple
    float_parameters: tuple
    instruction: str


def pipeline_for(manifest: Path):
    kind = json.loads(manifest.read_text()).get('pipeline', 'pcb-pair')
    if kind == 'pcb-pair':
        return Pipeline(kind, 'PCB defect localization', Config, Evaluator, detect, fingerprint,
                        (Config(scale=.5), Config(registration_scale=.5), Config(workers=2), Config(threads=2),
                         Config(registration_scale=.25), Config(workers=4), Config(threads=4), Config(threshold=100),
                         Config(min_area=16), Config(threshold=120), Config(scale=.75, workers=2), Config(min_area=4)),
                        ('scale', 'registration_scale'), '')
    if kind == 'pedestrian-yolox':
        from .pedestrian import PedestrianConfig as C, PedestrianEvaluator, detect_people, pedestrian_fingerprint
        return Pipeline(kind, 'Pedestrian localization', C, PedestrianEvaluator, detect_people, pedestrian_fingerprint,
                        (C(input_size=160), C(workers=4), C(input_size=224), C(confidence=.5), C(workers=2),
                         C(threads=2), C(threads=4), C(confidence=.7), C(confidence=.2), C(input_size=416),
                         C(input_size=224, workers=4), C(input_size=160, workers=4)),
                        ('confidence',),
                        'Single grayscale image converted to BGR. OpenCV Zoo YOLOX ONNX model, OpenCV DNN '
                        'CPU backend, person class only. Allowed input_size [160,224,320,416], '
                        'confidence [0.2,0.3,0.5,0.7], workers [1,2,4], threads [1,2,4]. No nested thread '
                        'pools. Lower input resolution reduces work but can lose people. Larger confidence '
                        'filters predictions and can reduce recall. Workers/threads cannot repair lost recall.')
    raise ValueError('unsupported pipeline')
