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
