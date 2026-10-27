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
