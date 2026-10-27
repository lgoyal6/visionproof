import base64
import copy
import io
import json
import struct
import time
import zipfile
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pytest

from opencv_deploy_proof import live, pedestrian
from opencv_deploy_proof.data import load_manifest


def archive(value, extra=None, suffix='.png'):
    image = np.zeros((12, 16), np.uint8)
    image[2:8, 3:9] = 255
    ok, data = cv2.imencode(suffix, image)
    assert ok
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('annotations.json', json.dumps(value))
        z.writestr('images/a.png', data.tobytes())
        if extra:
            z.writestr(extra, 'invalid')
    return output.getvalue()



@pytest.fixture
def coco():
    return {'images': [{'id': 1, 'file_name': 'images/a.png', 'width': 16, 'height': 12}],
            'categories': [{'id': 1, 'name': 'person'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [3, 2, 6, 6]}]}



@pytest.fixture
def imported(coco, tmp_path, monkeypatch):
    model = tmp_path / 'models'
    model.mkdir()
    for name in pedestrian.MODEL_HASHES:
        (model / name).write_bytes(b'explicit mocked detector fixture')
    monkeypatch.setattr(live, 'verify_models', lambda root: None)
    monkeypatch.setattr(pedestrian, 'verify_models', lambda root: None)
    identifier = live.import_coco(archive(coco), tmp_path, model, 'Own test set')
    manifest = tmp_path / 'imports' / identifier / 'manifest.json'
    return tmp_path, model, identifier, manifest



def test_import_normalizes_coco_and_preserves_original_input(imported, coco):
    root, _, identifier, manifest = imported
    value = load_manifest(manifest)
    assert value['private_upload'] and value['name'] == 'Own test set'
    row = value['records'][0]
    assert row['split'] == 'test' and row['group'] == 'user-import'
    assert (manifest.parent / row['label']).read_text() == '3 2 9 8 1'
    assert (manifest.parent / 'upload.zip').is_file()
    assert not (root / 'images').exists()



@pytest.mark.parametrize('change', ['outside', 'crowd', 'duplicate', 'unknown-image', 'category', 'no-person', 'dimensions'])
def test_import_refuses_invalid_annotations(imported, coco, change):
    root, model, _, _ = imported
    if change == 'outside': coco['annotations'][0]['bbox'][0] = -1
    if change == 'crowd': coco['annotations'][0]['iscrowd'] = 1
    if change == 'duplicate': coco['images'].append(copy.deepcopy(coco['images'][0]))
    if change == 'unknown-image': coco['annotations'][0]['image_id'] = 999
    if change == 'category': coco['annotations'][0]['category_id'] = 999
    if change == 'no-person': coco['categories'][0]['name'] = 'car'
    if change == 'dimensions': coco['images'][0]['width'] = 15
    before = set((root / 'imports').iterdir())
    with pytest.raises(ValueError):
        live.import_coco(archive(coco), root, model, 'invalid')
    assert set((root / 'imports').iterdir()) == before



def test_import_rejects_path_traversal_and_invalid_archive(imported, coco):
    root, model, _, _ = imported
    with pytest.raises(ValueError, match='unsafe'):
        live.import_coco(archive(coco, '../escape.txt'), root, model, 'bad')
    with pytest.raises(ValueError, match='ZIP'):
        live.import_coco(b'not a zip', root, model, 'bad')
    assert not (root.parent / 'escape.txt').exists()



def test_images_are_bounded_before_decode_and_jpeg_is_supported():
    image = np.zeros((12, 16), np.uint8)
    for extension in ('.png', '.jpg'):
        _, raw = cv2.imencode(extension, image)
        assert live.decode_image(raw.tobytes()).shape == (12, 16)
    huge = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 100000, 100000)
    with pytest.raises(ValueError, match='megapixels'):
        live.decode_image(huge)
    with pytest.raises(ValueError):
        live.decode_base64('invalid===', 100)
