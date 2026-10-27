"""Local image comparisons and bounded COCO person-dataset imports."""

import base64
import binascii
import hashlib
import io
import json
import math
import shutil
import struct
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from time import perf_counter

import cv2
import numpy as np

from .comparison import compare, signed, verify_comparison
from .data import digest, load_manifest
from .evaluate import Requirements, match
from .pedestrian import MODEL_COMMIT, MODEL_HASHES, PedestrianConfig, detect_people, verify_models

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 4_194_304


def decode_image(raw):
    """Check common image headers before allowing OpenCV to allocate pixels."""
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('image must be a PNG or JPEG smaller than 8 MiB')
    width = height = 0
    if raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw) >= 24 and raw[12:16] == b'IHDR':
        width, height = struct.unpack('>II', raw[16:24])
    elif raw.startswith(b'\xff\xd8'):
        offset = 2
        while offset + 4 <= len(raw):
            if raw[offset] != 255:
                break
            while offset < len(raw) and raw[offset] == 255:
                offset += 1
            if offset + 3 > len(raw):
                break
            marker = raw[offset]
            size = int.from_bytes(raw[offset + 1:offset + 3], 'big')
            if marker in (0xC0, 0xC1, 0xC2) and size >= 8 and offset + 8 <= len(raw):
                height, width = struct.unpack('>HH', raw[offset + 4:offset + 8])
                break
            if size < 2:
                break
            offset += size + 1
    if not (0 < width <= 8192 and 0 < height <= 8192 and width * height <= MAX_PIXELS):
        raise ValueError('use a valid PNG or JPEG of at most 4 megapixels')
    try:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    except cv2.error:
        raise ValueError('image could not be decoded') from None
    if image is None or image.shape != (height, width):
        raise ValueError('image could not be decoded at its declared dimensions')
    return image


def decode_base64(value, limit):
    if not isinstance(value, str) or len(value) > ((limit + 2) // 3) * 4:
        raise ValueError('upload exceeds the size limit')
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError('upload is not valid base64') from None
    if len(raw) > limit:
        raise ValueError('upload exceeds the size limit')
    return raw


def configuration(value):
    if not isinstance(value, dict) or set(value) - {'input_size', 'confidence', 'workers', 'threads'}:
        raise ValueError('unknown detector configuration')
    if any(type(value[k]) is not int for k in ('input_size', 'workers', 'threads') if k in value):
        raise ValueError('input size, workers and threads must be integers')
    return PedestrianConfig(**value)


def requirements(value):
    if not isinstance(value, dict) or set(value) != {'max_recall_drop', 'max_precision_drop', 'max_p95_ratio'}:
        raise ValueError('declare recall, precision and latency limits')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values()):
        raise ValueError('limits must be finite numbers')
    if not (0 <= value['max_recall_drop'] <= .2 and 0 <= value['max_precision_drop'] <= .2
            and 1 <= value['max_p95_ratio'] <= 2):
        raise ValueError('accuracy drops must be 0..20 percentage points; latency ratio must be 1..2')
    return Requirements(**value)


def import_coco(raw, root: Path, model_root: Path, name):
    """Normalize selected COCO person images without extracting archive paths."""
    if not raw or len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError('dataset ZIP must be smaller than 32 MiB')
    verify_models(model_root)
    try:
        zipped = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise ValueError('dataset must be a ZIP archive') from None
    with zipped:
        members = zipped.infolist()
        if len(members) > 160 or sum(x.file_size for x in members) > 64 * 1024 * 1024:
            raise ValueError('archive exceeds 160 entries or 64 MiB expanded')
        paths = [PurePosixPath(x.filename) for x in members]
        if (len(set(x.filename for x in members)) != len(members)
                or any(p.is_absolute() or '..' in p.parts or '\\' in str(p) for p in paths)
                or any(x.flag_bits & 1 or (x.external_attr >> 16) & 0o170000 == 0o120000 for x in members)):
            raise ValueError('unsafe, duplicate or encrypted archive entries')
        if 'annotations.json' not in zipped.namelist():
            raise ValueError('ZIP must contain annotations.json at its root')
        if zipped.getinfo('annotations.json').file_size > 2 * 1024 * 1024:
            raise ValueError('annotation file exceeds 2 MiB')
        try:
            coco = json.loads(zipped.read('annotations.json'))
            images, annotations, categories = (coco[k] for k in ('images', 'annotations', 'categories'))
        except (ValueError, KeyError, TypeError):
            raise ValueError('annotations.json needs COCO images, annotations and categories') from None
        if not (isinstance(images, list) and 1 <= len(images) <= 64
                and isinstance(annotations, list) and len(annotations) <= 10000 and isinstance(categories, list)):
            raise ValueError('import 1..64 images and at most 10000 annotations')
        if any(not isinstance(c, dict) or type(c.get('id')) is not int or not isinstance(c.get('name'), str) for c in categories):
            raise ValueError('categories need integer IDs and names')
        category_ids = {c['id'] for c in categories}
        if len(category_ids) != len(categories):
            raise ValueError('category IDs must be unique')
        people = {c['id'] for c in categories if c['name'] == 'person'}
        if len(people) != 1:
            raise ValueError('declare exactly one category named person')
        ids, files, normalized, total_pixels = set(), set(), [], 0
        for row in images:
            if not isinstance(row, dict) or type(row.get('id')) is not int or row['id'] in ids:
                raise ValueError('image IDs must be unique integers')
            file = row.get('file_name')
            if not isinstance(file, str) or file in files or file not in zipped.namelist():
                raise ValueError('each image needs a unique existing file_name')
            if zipped.getinfo(file).file_size > MAX_IMAGE_BYTES:
                raise ValueError('each image must be smaller than 8 MiB')
            image = decode_image(zipped.read(file))
            if row.get('width') != image.shape[1] or row.get('height') != image.shape[0]:
                raise ValueError('COCO width and height must match the image pixels')
            total_pixels += image.size
            if total_pixels > 32 * 1024 * 1024:
                raise ValueError('dataset exceeds 32 megapixels in total')
            ids.add(row['id']); files.add(file)
            normalized.append((row, image, []))
        by_id = {r['id']: (r, image, boxes) for r, image, boxes in normalized}
        annotation_ids = set()
        for annotation in annotations:
            if not isinstance(annotation, dict) or type(annotation.get('id')) is not int or annotation['id'] in annotation_ids:
                raise ValueError('annotation IDs must be unique integers')
            annotation_ids.add(annotation['id'])
            if type(annotation.get('image_id')) is not int or annotation['image_id'] not in by_id:
                raise ValueError('annotation refers to an unknown image')
            if type(annotation.get('category_id')) is not int or annotation['category_id'] not in category_ids:
                raise ValueError('annotation refers to an unknown category')
            if annotation.get('category_id') not in people:
                continue
            if annotation.get('iscrowd', 0) != 0:
                raise ValueError('crowd annotations are not supported by this scorer')
            row, image, boxes = by_id[annotation['image_id']]
            box = annotation.get('bbox')
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in box)):
                raise ValueError('each person bbox must be finite [x, y, width, height]')
            x, y, width, height = box
            if not (0 <= x < x + width <= image.shape[1] and 0 <= y < y + height <= image.shape[0]):
                raise ValueError('person bounding box is outside the image')
            box = [round(x), round(y), round(x + width), round(y + height), 1]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('person boxes must cover at least one pixel')
            boxes.append(box)
        if not any(boxes for _, _, boxes in normalized):
            raise ValueError('dataset must contain at least one labeled person')
    identifier = 'import-' + uuid.uuid4().hex[:12]
    directory = root / 'imports' / identifier
    directory.mkdir(parents=True)
    try:
        (directory / 'models').mkdir()
        for file in MODEL_HASHES:
            shutil.copy2(model_root / file, directory / 'models' / file)
        records = []
        for index, (row, image, boxes) in enumerate(normalized):
            path, label = directory / f'{index:03}.png', directory / f'{index:03}.txt'
            if not cv2.imwrite(str(path), image):
                raise ValueError('could not save normalized image')
            label.write_text('\n'.join(' '.join(map(str, b)) for b in boxes))
            records.append({'id': str(row['id']), 'split': 'test', 'group': 'user-import',
                            'image': path.name, 'template': path.name, 'label': label.name,
                            'sha256': {'image': digest(path), 'template': digest(path), 'label': digest(label)}})
        archive_sha = hashlib.sha256(raw).hexdigest()
        manifest = signed({'pipeline': 'pedestrian-yolox', 'source': 'user-upload', 'commit': archive_sha,
                           'name': str(name)[:80], 'private_upload': True,
                           'model': {'commit': MODEL_COMMIT, 'sha256': MODEL_HASHES}, 'records': records,
                           'split_method': 'User-supplied test set for fixed-configuration comparison; no search.',
                           'limitations': ['User annotations are not independently verified.',
                                           'Imported images are one source group; no group independence is established.',
                                           'Repeated evaluation is not a new untouched holdout.']})
        (directory / 'upload.zip').write_bytes(raw)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return identifier
    except Exception:
        shutil.rmtree(directory)
        raise


class LiveWorkbench:
    def __init__(self, root, registered, execution_lock):
        self.root, self.registered, self.execution_lock = root, registered, execution_lock
        self.lock, self.jobs = Lock(), {}

    def models(self):
        for manifest in self.registered().values():
            model_root = manifest.parent / 'models'
            verify_models(model_root)
            return model_root
        raise ValueError('a qualified pedestrian run and pinned YOLOX model are required')

    def datasets(self):
        values = dict(self.registered())
        values.update({p.parent.name: p for p in (self.root / 'imports').glob('import-*/manifest.json')})
        return values

    def dataset(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.datasets():
            raise ValueError('unknown dataset')
        path = self.datasets()[identifier]
        value = load_manifest(path)
        return path, value, [r for r in value['records'] if r['split'] == 'test']

    def catalog(self):
        summaries = []
        for identifier in self.datasets():
            _, value, rows = self.dataset(identifier)
            summaries.append({'id': identifier, 'name': value.get('name') or identifier,
                              'image_count': len(rows), 'private': bool(value.get('private_upload'))})
        return {'datasets': summaries, 'pipeline': 'pedestrian-yolox'}

    def images(self, identifier):
        path, _, rows = self.dataset(identifier)
        return {'images': [{'id': r['id'], 'labels': [[int(x) for x in line.split()] for line in
                           (path.parent / r['label']).read_text().splitlines() if line.strip()]} for r in rows]}

    def image(self, identifier, image_id):
        path, _, rows = self.dataset(identifier)
        row = next((r for r in rows if r['id'] == image_id), None)
        if row is None:
            raise ValueError('unknown image')
        return (path.parent / row['image']).read_bytes()

    def preview(self, value):
        if set(value) - {'dataset', 'id', 'frame', 'baseline', 'candidate'}:
            raise ValueError('unknown comparison option')
        baseline, candidate = configuration(value.get('baseline', {})), configuration(value.get('candidate', {}))
        labels = None
        if 'frame' in value:
            if 'dataset' in value or 'id' in value:
                raise ValueError('choose a frame or a dataset image')
            image = decode_image(decode_base64(value['frame'], MAX_IMAGE_BYTES))
            models = self.models()
        else:
            path, _, _ = self.dataset(value.get('dataset'))
            image = decode_image(self.image(value['dataset'], value.get('id')))
            labels = next(r['labels'] for r in self.images(value['dataset'])['images'] if r['id'] == value['id'])
            models = path.parent / 'models'
        if not self.execution_lock.acquire(blocking=False):
            raise RuntimeError('another evaluation is running; try again after it finishes')
        try:
            results = {}
            for name, config in [('baseline', baseline), ('candidate', candidate)]:
                cv2.setNumThreads(config.threads)
                detect_people(image, models, config)
                start = perf_counter()
                output = detect_people(image, models, config)
                results[name] = {'config': config.to_dict(), 'boxes': output['boxes'],
                                 'warm_image_ms': (perf_counter() - start) * 1000,
                                 'quality': match(output['boxes'], labels) if labels is not None else None}
        finally:
            self.execution_lock.release()
        left, right = results['baseline'], results['candidate']
        delta = {'baseline_only_predictions': match(right['boxes'], [b + [1] for b in left['boxes']])['fn']}
        if labels is not None:
            delta['newly_missed_annotations'] = sorted(set(right['quality']['missed_indices']) - set(left['quality']['missed_indices']))
        return {'kind': 'live-image-comparison', 'shape': list(image.shape), 'labels': labels,
                'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(), **results, 'delta': delta,
                'scope': 'Single warm image preview, not a batch latency qualification. Workers apply to batches. '
                         + ('Labels available for accuracy scoring.' if labels is not None else
                            'No labels: prediction differences do not establish accuracy. Frames are not saved.')}

    def start(self, value):
        if set(value) != {'dataset', 'baseline', 'candidate', 'requirements'}:
            raise ValueError('declare dataset, two configurations and quality limits')
        manifest, _, _ = self.dataset(value['dataset'])
        configs = {name: configuration(value[name]).to_dict() for name in ('baseline', 'candidate')}
        limits = requirements(value['requirements'])
        if not self.execution_lock.acquire(blocking=False):
            raise RuntimeError('another evaluation is running; try again after it finishes')
        identifier = 'live-check-' + uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[identifier] = {'id': identifier, 'status': 'running', 'dataset': value['dataset']}

        def run():
            try:
                result = compare(manifest, configs, self.root / identifier, rounds=3, repeats=3,
                                 provenance={'private_upload': True, 'mode': 'user-fixed-configurations'}, requirements=limits)
                with self.lock:
                    self.jobs[identifier].update(status='completed', passed=result['rows'][1]['gate']['passed'],
                                                 report=result)
            except Exception as error:
                with self.lock:
                    self.jobs[identifier].update(status='failed', error=str(error) if isinstance(error, ValueError) else type(error).__name__)
            finally:
                self.execution_lock.release()
        try:
            Thread(target=run, daemon=True).start()
        except Exception:
            self.execution_lock.release()
            raise
        return self.status(identifier)

    def status(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                # Finished comparisons remain usable after a server restart.
                if not isinstance(identifier, str) or not identifier.startswith('live-check-') or not identifier[11:].isalnum():
                    raise ValueError('unknown evaluation')
                if not (self.root / identifier / 'comparison.json').is_file():
                    raise ValueError('unknown or incomplete evaluation')
                result = verify_comparison(self.root / identifier)['report']
                return {'id': identifier, 'status': 'completed', 'passed': result['rows'][1]['gate']['passed'], 'report': result}
            return dict(self.jobs[identifier])
