"""A second, single-image pipeline using OpenCV Zoo YOLOX on the CPU DNN backend.

YOLOX decoding follows OpenCV Zoo's Apache-2.0 reference implementation.
"""

import json
import shutil
import statistics
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import local
from time import perf_counter

import cv2
import numpy as np

from .data import canonical_digest, digest, load_manifest
from .evaluate import fingerprint, match, metrics

SOURCE = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
# Pinned to the original authors' archive, not a moving dataset release.
ARCHIVE_SHA256 = '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2'
MODEL_COMMIT = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
MODEL_HASHES = {
    'object_detection_yolox_2022nov.onnx': 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063',
}


@dataclass(frozen=True)
class PedestrianConfig:
    input_size: int = 320
    confidence: float = 0.3
    workers: int = 1
    threads: int = 1

    def __post_init__(self):
        if (type(self.input_size) is not int or self.input_size not in (160, 224, 320, 416)
                or type(self.confidence) not in (float, int) or self.confidence not in (0.2, 0.3, 0.5, 0.7)
                or self.workers not in (1, 2, 4) or self.threads not in (1, 2, 4)
                or self.workers > 1 and self.threads > 1):
            raise ValueError('invalid YOLOX configuration or nested thread pools')

    def to_dict(self):
        return asdict(self)


_thread = local()


def verify_models(root):
    if any(not (root / name).is_file() or digest(root / name) != sha for name, sha in MODEL_HASHES.items()):
        raise ValueError('YOLOX files do not match the pinned model')


def detect_people(image, model_root, config):
    if image is None or image.ndim != 2:
        raise ValueError('pedestrian detector expects a grayscale image')
    model_root = Path(model_root)
    if getattr(_thread, 'model_root', None) != model_root:
        verify_models(model_root)
        _thread.net = cv2.dnn.readNetFromONNX(str(model_root / 'object_detection_yolox_2022nov.onnx'))
        _thread.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _thread.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _thread.model_root = model_root
    start = perf_counter()
    height, width = image.shape
    size = config.input_size
    ratio = size / max(height, width)
    resized = cv2.resize(image, (max(1, int(width * ratio)), max(1, int(height * ratio))))
    padded = np.full((size, size), 114, np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), swapRB=False)
    _thread.net.setInput(blob)
    detections = _thread.net.forward()[0].copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        x, y = np.meshgrid(np.arange(size // stride), np.arange(size // stride))
        grid = np.stack((x, y), axis=2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((len(grid), 1), stride))
    strides = np.concatenate(strides)
    if len(detections) != len(strides):
        raise ValueError('YOLOX output does not match declared input resolution')
    detections[:, :2] = (detections[:, :2] + np.concatenate(grids)) * strides
    detections[:, 2:4] = np.exp(detections[:, 2:4]) * strides
    scores = detections[:, 4] * detections[:, 5]  # COCO person class 0.
    chosen = scores >= config.confidence
    xywh = detections[chosen, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    xywh /= ratio
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores[chosen].tolist(), config.confidence, .5)
    boxes = []
    for index in np.asarray(keep).reshape(-1):
        x1, y1, w, h = xywh[int(index)]
        x2, y2 = x1 + w, y1 + h
        box = [max(0, round(x1)), max(0, round(y1)), min(width, round(x2)), min(height, round(y2))]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return {'boxes': sorted(boxes), 'stages_ms': {'dnn_forward': (perf_counter() - start) * 1000}}


def pedestrian_fingerprint():
    result = fingerprint()
    result['code_sha256']['pedestrian.py'] = digest(Path(__file__))
    result['model_sha256'] = MODEL_HASHES
    return result


def prepare_pedestrians(root: Path, limit=48):
    if type(limit) is not int or not 1 <= limit <= 70:
        raise ValueError('pedestrian sample size must be between 1 and 70 per split')
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'source.json'
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix='visionproof-penn-') as temp:
            archive = Path(temp) / 'dataset.zip'
            with urllib.request.urlopen(SOURCE, timeout=90) as response, archive.open('wb') as out:
                shutil.copyfileobj(response, out)
            if digest(archive) != ARCHIVE_SHA256:
                raise ValueError('Penn-Fudan archive differs from the pinned download')
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    if member.is_dir():
                        continue
                    target = (root / member.filename).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError('unsafe dataset archive path')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zipped.read(member))
            marker.write_text(json.dumps({'source': SOURCE, 'archive_sha256': ARCHIVE_SHA256}, indent=2))
    if json.loads(marker.read_text()) != {'source': SOURCE, 'archive_sha256': ARCHIVE_SHA256}:
        raise ValueError('Penn-Fudan source marker does not match the pinned archive')
    models = root / 'models'
    models.mkdir(exist_ok=True)
    for name, sha in MODEL_HASHES.items():
        target = models / name
        if not target.exists():
            url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_COMMIT}/models/object_detection_yolox/{name}'
            with urllib.request.urlopen(url, timeout=90) as response, target.open('wb') as out:
                shutil.copyfileobj(response, out)
        if digest(target) != sha:
            raise ValueError('model download does not match pinned hash')
    path = root / 'manifest.json'
    if path.exists():
        manifest = load_manifest(path)
        if manifest.get('sample_limit') != limit:
            raise ValueError('use a new data directory to declare a different sample size')
        return path
    records = []
    label_root = root / 'derived-boxes'
    label_root.mkdir(exist_ok=True)
    # Source-location separation is predeclared, not chosen from detector outcomes.
    for split, prefix in [('development', 'FudanPed'), ('test', 'PennPed')]:
        files = sorted((root / 'PennFudanPed/PNGImages').glob(prefix + '*.png'),
                       key=lambda p: canonical_digest('visionproof-dnn-v1:' + p.stem))[:limit]
        if len(files) != limit:
            raise ValueError('incomplete pedestrian dataset')
        for image in files:
            mask_path = root / 'PennFudanPed/PedMasks' / (image.stem + '_mask.png')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise ValueError('invalid pedestrian instance mask')
            boxes = []
            for identity in np.unique(mask):
                if identity == 0:
                    continue
                ys, xs = np.where(mask == identity)
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, 1])
            label = label_root / (image.stem + '.txt')
            label.write_text('\n'.join(' '.join(map(str, box)) for box in boxes) + '\n')
            paths = {'image': image, 'template': image, 'label': label}
            records.append({'id': image.stem, 'group': prefix, 'split': split,
                            **{k: str(p.relative_to(root)) for k, p in paths.items()},
                            'sha256': {k: digest(p) for k, p in paths.items()},
                            'mask_sha256': digest(mask_path)})
    manifest = {'pipeline': 'pedestrian-yolox', 'source': SOURCE, 'commit': ARCHIVE_SHA256,
                'model': {'source': 'https://github.com/opencv/opencv_zoo/tree/' + MODEL_COMMIT + '/models/object_detection_yolox', 'commit': MODEL_COMMIT,
                          'sha256': MODEL_HASHES, 'backend': 'OpenCV DNN / CPU', 'input': 'grayscale replicated to BGR'},
                'sample_limit': limit, 'records': records,
                'split_method': 'Predeclared Fudan development / Penn test; SHA256-ordered images.',
                'limitations': ['Penn-Fudan is a small public research dataset, not deployment validation.',
                                'Source-location split does not guarantee person-identity separation.',
                                'Bounding boxes are derived from public instance masks; no defect classes.']}
    manifest['sha256'] = canonical_digest(manifest)
    path.write_text(json.dumps(manifest, indent=2))
    return path


class PedestrianEvaluator:
    def __init__(self, manifest: Path, split: str, repeats=9):
        if split not in ('development', 'test') or type(repeats) is not int or repeats < 3:
            raise ValueError('invalid split or repeat count')
        self.manifest = load_manifest(manifest)
        if self.manifest.get('pipeline') != 'pedestrian-yolox' or self.manifest.get('model', {}).get('sha256') != MODEL_HASHES:
            raise ValueError('expected a pedestrian manifest')
        self.split, self.repeats = split, repeats
        self.model_root = manifest.parent / 'models'
        verify_models(self.model_root)
        self.rows = []
        for row in self.manifest['records']:
            if row['split'] == split:
                image = cv2.imread(str(manifest.parent / row['image']), 0)
                labels = [[int(v) for v in line.split()] for line in
                          (manifest.parent / row['label']).read_text().splitlines() if line.strip()]
                if image is None or any(len(label) != 5 for label in labels):
                    raise ValueError('invalid pedestrian example')
                self.rows.append((row, image, labels))
        if not self.rows:
            raise ValueError('empty pedestrian split')

    def run(self, config: PedestrianConfig):
        cv2.setNumThreads(config.threads)
        durations = []
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            for iteration in range(self.repeats + 2):
                start = perf_counter()
                def execute(row):
                    return detect_people(row[1], self.model_root, config)
                outputs = (list(pool.map(execute, self.rows)) if config.workers > 1
                           else [execute(row) for row in self.rows])
                if iteration >= 2:
                    durations.append((perf_counter() - start) * 1000)
        totals = {'tp': 0, 'fp': 0, 'fn': 0}
        examples = []
        for (row, image, labels), output in zip(self.rows, outputs):
            scored = match(output['boxes'], labels)
            for key in totals:
                totals[key] += scored[key]
            examples.append({'id': row['id'], 'image': row['image'], 'shape': list(image.shape),
                             'labels': labels, 'predictions': output['boxes'], **scored})
        median = statistics.median(durations)
        result = {'pipeline': 'pedestrian-yolox', 'config': config.to_dict(), 'split': self.split,
                  'manifest_sha256': self.manifest['sha256'], 'quality': metrics(**totals),
                  'per_class': {'1': {'total': totals['tp'] + totals['fn'], 'missed': totals['fn']}},
                  'examples': examples,
                  'performance': {'batch_ms': durations, 'median_batch_ms': median,
                                  'p95_batch_ms': float(np.percentile(durations, 95)),
                                  'images_per_second': len(self.rows) * 1000 / median,
                                  'image_count': len(self.rows), 'repeats': self.repeats,
                                  'scope': 'warm in-memory batch; excludes decode, networking, cold start and service queueing'},
                  'stages_ms': {'dnn_forward': statistics.median(o['stages_ms']['dnn_forward'] for o in outputs)},
                  'environment': pedestrian_fingerprint()}
        result['sha256'] = canonical_digest(result)
        return result
