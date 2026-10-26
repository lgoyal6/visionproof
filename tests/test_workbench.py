import copy
import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_deploy_proof.data import canonical_digest, digest
from opencv_deploy_proof.diagnostics import paired_diagnostics
from opencv_deploy_proof.evidence import export_bundle, verify_run
from opencv_deploy_proof.experiment import proof
from opencv_deploy_proof.journal import append_event, read_events
from opencv_deploy_proof.pipeline import Config
from opencv_deploy_proof.policy import materialize
from opencv_deploy_proof.serve import create_server


@pytest.fixture(scope='module')
def qualified(tmp_path_factory):
    root = tmp_path_factory.mktemp('qualification')
    records = []
    for split in ['development', 'test']:
        image = np.zeros((96, 96), np.uint8)
        image[30:50, 30:50] = 255
        reference = np.zeros_like(image)
        paths = {'image': root / (split + '.png'), 'template': root / (split + '-ref.png'),
                 'label': root / (split + '.txt')}
        cv2.imwrite(str(paths['image']), image)
        cv2.imwrite(str(paths['template']), reference)
        paths['label'].write_text('22 22 58 58 1\n')
        records.append({'id': split, 'group': split, 'split': split,
                        **{k: p.name for k, p in paths.items()},
                        'sha256': {k: digest(p) for k, p in paths.items()}})
    manifest = {'records': records, 'source': 'synthetic test fixture', 'commit': 'fixture', 'limitations': []}
    manifest['sha256'] = canonical_digest(manifest)
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    # This fixture tests evidence and HTTP behavior, not microsecond wall-clock noise.
    # Preserve real OpenCV predictions while giving each batch a deterministic clock.
    from itertools import count
    clock = count()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('opencv_deploy_proof.evaluate.perf_counter', lambda: next(clock) / 1000)
        proof(path, root / 'run', provider='rules', budget=1, repeats=3)
    return root, path


def test_paired_diagnostics_identify_lost_and_recovered_annotations():
    before = {'manifest_sha256': 'x', 'split': 'development', 'config': Config().to_dict(),
              'stages_ms': {'registration': 9, 'contours': 1},
              'examples': [{'id': 'a', 'labels': [[0, 0, 10, 10, 1], [20, 20, 80, 80, 2]], 'missed_indices': [1]}]}
    after = copy.deepcopy(before)
    after['config']['scale'] = .5
    after['examples'][0]['missed_indices'] = [0]
    diagnostic = paired_diagnostics(before, after)
    assert diagnostic['newly_missed_count'] == diagnostic['recovered_count'] == 1
    assert diagnostic['newly_missed_examples'][0]['size'] == 'small'
    assert diagnostic['baseline_stage_share']['registration'] == .9
    assert diagnostic['size_buckets']['large']['candidate_missed'] == 0


def test_structured_proposal_cannot_change_another_parameter_or_reuse_config():
    history = [{'result': {'config': Config().to_dict()}}]
    proposal = {'base_observation': 0, 'parameter': 'workers', 'value': 2,
                'hypothesis': 'Independent images can be concurrent', 'falsification': 'p95 fails'}
    assert materialize(proposal, history) == Config(workers=2)
    with pytest.raises(ValueError):
        materialize({**proposal, 'base_observation': -1}, history)
    with pytest.raises(ValueError):
        materialize({**proposal, 'value': 1}, history)
    with pytest.raises(ValueError):
        materialize({**proposal, 'value': 1.5}, history)
    with pytest.raises(ValueError):
        materialize({**proposal, 'scale': .5}, history)


def test_journal_detects_modified_and_reordered_events(tmp_path):
    path = tmp_path / 'events.jsonl'
    append_event(path, 'start', {'budget': 4})
    append_event(path, 'completed', {'passed': False})
    assert len(read_events(path)) == 2
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(reversed(lines)))
    with pytest.raises(ValueError, match='journal verification'):
        read_events(path)


def test_fresh_protocol_excludes_prior_images_and_keeps_board_groups_disjoint(tmp_path):
    from opencv_deploy_proof.protocol import fresh_manifest
    from opencv_deploy_proof.data import load_manifest
    rows = []
    for group in ['group12100', 'group13000', 'group20085', 'group00041']:
        directory = tmp_path / 'PCBData' / group / 'board'
        labels = directory.with_name('board_not')
        directory.mkdir(parents=True)
        labels.mkdir()
        for index in range(3):
            identifier = group + '-' + str(index)
            paths = {'image': directory / (identifier + '_test.jpg'),
                     'template': directory / (identifier + '_temp.jpg'),
                     'label': labels / (identifier + '.txt')}
            for path in paths.values():
                path.write_text('hash fixture')
            rows.append({'id': identifier, 'group': group, 'split': 'calibration',
                         **{k: str(p.relative_to(tmp_path)) for k, p in paths.items()},
                         'sha256': {k: digest(p) for k, p in paths.items()}})
    original = {'source': 'fixture', 'commit': 'fixture', 'records': [rows[0]]}
    original['sha256'] = canonical_digest(original)
    (tmp_path / 'original.json').write_text(json.dumps(original))
    output = fresh_manifest(tmp_path / 'original.json', tmp_path / 'fresh.json', limit=4)
    manifest = load_manifest(output)
    assert len(manifest['records']) == 8
    assert rows[0]['id'] not in {r['id'] for r in manifest['records']}
    groups = [{r['group'] for r in manifest['records'] if r['split'] == split} for split in ('development', 'test')]
    assert groups[0].isdisjoint(groups[1])
    with pytest.raises(ValueError, match='overwrite'):
        fresh_manifest(tmp_path / 'original.json', output)


def test_selection_is_recorded_before_holdout_access(qualified):
    root, _ = qualified
    events = read_events(root / 'run/events.jsonl')
    kinds = [e['kind'] for e in events]
    assert kinds.index('selection_frozen') < kinds.index('holdout_started') < kinds.index('completed')
    selected = events[kinds.index('selection_frozen')]['payload']['sha256']
    assert events[kinds.index('holdout_started')]['payload']['selection_sha256'] == selected


def test_export_binds_config_and_includes_source_and_scoring_inputs(qualified):
    root, path = qualified
    item = verify_run(root / 'run', path, require_pass=True)
    with zipfile.ZipFile(io.BytesIO(export_bundle(item))) as archive:
        assert json.loads(archive.read('selected-config.json')) == item['selection']['config']
        assert 'src/opencv_deploy_proof/pipeline.py' in archive.namelist()
        assert json.loads(archive.read('manifest.json'))['sha256'] == item['manifest']['sha256']


def test_public_replay_exports_only_static_sanitized_routes(qualified, tmp_path):
    from opencv_deploy_proof.public_demo import export_public
    root, _ = qualified
    out = tmp_path / 'public'
    exported = export_public(root, out, ['run'], 'run')
    assert exported['runs'] == 1
    config = (out / 'public-config.js').read_text()
    routes = json.loads(config.removeprefix('window.VISIONPROOF_PUBLIC = ').removesuffix(';\n'))['routes']
    assert all((out / target.lstrip('/')).is_file() for target in routes.values())
    assert '/api/jobs' not in routes and not any('inspect?' in key for key in routes)
    view = json.loads((out / routes['/api/run?run=run'].lstrip('/')).read_text())
    assert view['quality'] == verify_run(root / 'run')['selected']['quality']
    assert 'manifest_path' not in json.dumps(view)
    assert str(root) not in json.dumps(view)
    assert 'public-config.js' in (out / 'index.html').read_text()


def test_rehashed_verdict_cannot_point_to_a_different_selection(qualified, tmp_path):
    import shutil
    root, path = qualified
    shutil.copytree(root / 'run', tmp_path / 'changed')
    target = tmp_path / 'changed/verdict.json'
    value = json.loads(target.read_text())
    value['selection']['config']['workers'] = 4
    value.pop('sha256')
    value['sha256'] = canonical_digest(value)
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='mismatch'):
        verify_run(tmp_path / 'changed', path)


@pytest.fixture
def http_server(qualified):
    root, path = qualified
    server = create_server(path, root / 'run', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}', server
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_reads_actual_measurements_and_refuses_unknown_paths(http_server):
    base, _ = http_server
    for route in ['/', '/app.js', '/api/catalog', '/api/run', '/api/pixels?id=test', '/api/predict?id=test']:
        with urllib.request.urlopen(base + route) as response:
            assert response.status == 200
            assert response.headers['X-Content-Type-Options'] == 'nosniff'
    for route in ['/api/predict?id=unknown', '/api/run?run=../../etc', '/static/../../etc/passwd']:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + route)
        assert error.value.code == 404


def test_mutation_requires_same_origin_and_valid_bounded_options(http_server):
    base, _ = http_server
    for origin, payload, expected in [('https://evil.invalid', {}, 403),
                                      (base, {'provider': 'rules', 'budget': 1000}, 400),
                                      (base, {'provider': 'shell', 'command': 'id'}, 400)]:
        request = urllib.request.Request(base + '/api/jobs', data=json.dumps(payload).encode(),
                                         headers={'Origin': origin, 'Content-Type': 'application/json'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == expected


def test_job_manager_excludes_overlapping_benchmarks(qualified, monkeypatch):
    from opencv_deploy_proof import jobs
    root, path = qualified
    entered, release = threading.Event(), threading.Event()

    def block(manifest, out, *args, **kwargs):
        out.mkdir()
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(jobs, 'proof', block)
    manager = jobs.Jobs(root, path, None)
    job = manager.start({'provider': 'rules', 'budget': 1})
    assert entered.wait(timeout=2)
    try:
        with pytest.raises(RuntimeError, match='already running'):
            manager.start({'provider': 'rules', 'budget': 1})
        assert manager.status(job['id'])['status'] == 'running'
    finally:
        release.set()
