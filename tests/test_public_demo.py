import json
from pathlib import Path

import pytest

from opencv_deploy_proof.public_demo import export_public, sanitize


def test_sanitizer_removes_paths_resources_and_private_fields():
    value = {'manifest_path': '/Users/person/data.json', 'prompt': 'private',
             'message': 'module /home/ubuntu/pkg/file.py i-0123456789abcdef0 123456789012',
             'quality': {'tp': 138}, 'url': 'https://github.com/opencv/opencv_zoo'}
    clean = sanitize(value)
    assert 'manifest_path' not in clean and 'prompt' not in clean
    assert '/home/' not in clean['message'] and 'i-0123' not in clean['message']
    assert '123456789012' not in clean['message']
    assert clean['quality'] == {'tp': 138}
    assert clean['url'] == value['url']


def test_public_export_requires_explicit_default_and_safe_run_names(tmp_path):
    with pytest.raises(ValueError, match='default'):
        export_public(tmp_path, tmp_path / 'out', [], 'unknown')
    with pytest.raises(ValueError, match='simple'):
        export_public(tmp_path, tmp_path / 'out', ['../secret'], '../secret')
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'keep').write_text('evidence')
    with pytest.raises(ValueError, match='empty'):
        export_public(tmp_path, out, ['run'], 'run')
