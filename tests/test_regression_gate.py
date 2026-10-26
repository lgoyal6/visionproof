import json

import cv2
import numpy as np
import pytest

from opencv_deploy_proof.comparison import compare, interval, verify_comparison
from opencv_deploy_proof.data import canonical_digest, digest
from opencv_deploy_proof.evaluate import Evaluator
from opencv_deploy_proof.pipeline import Config


@pytest.fixture
def regression_manifest(tmp_path):
    reference = np.zeros((192, 192), np.uint8)
    image = reference.copy()
    image[40:45, 40:45] = 255
    paths = {'image': tmp_path / 'image.png', 'template': tmp_path / 'reference.png',
             'label': tmp_path / 'labels.txt'}
    cv2.imwrite(str(paths['image']), image)
    cv2.imwrite(str(paths['template']), reference)
    paths['label'].write_text('34 34 51 51 1\n')
    manifest = {'records': [{'id': 'small-object', 'split': 'test', 'group': 'reserved-board',
                            **{k: p.name for k, p in paths.items()},
                            'sha256': {k: digest(p) for k, p in paths.items()}}],
                'source': 'synthetic regression fixture', 'commit': 'fixture', 'limitations': []}
    manifest['sha256'] = canonical_digest(manifest)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def stable_timings(monkeypatch):
    original = Evaluator.run

    def measured(self, config):
        result = original(self, config)
        # Quality and boxes are real OpenCV outputs; clock noise is not this test's subject.
        duration = 1 if config.scale == .5 else 2
        result['performance'].update(batch_ms=[duration] * self.repeats, median_batch_ms=duration,
                                     p95_batch_ms=duration, images_per_second=1000 / duration)
        result.pop('sha256')
        result['sha256'] = canonical_digest(result)
        return result

    monkeypatch.setattr(Evaluator, 'run', measured)


def test_real_small_object_regression_is_faster_but_blocked(regression_manifest, stable_timings, tmp_path):
    report = compare(regression_manifest, {'baseline': {}, 'fast': {'scale': .5}},
                     tmp_path / 'comparison', rounds=3, repeats=3)
    baseline, fast = report['rows']
    assert baseline['quality']['recall'] == 1
    assert fast['quality']['recall'] == 0
    assert fast['paired_speedup']['median'] == 2
    assert not fast['gate']['passed'] and not fast['gate']['checks']['recall']
    assert fast['lost_count'] == 1
    assert verify_comparison(tmp_path / 'comparison')['report'] == report
    contract = json.loads((tmp_path / 'comparison/comparison-contract.json').read_text())
    assert sorted(contract['order'][0]) == ['baseline', 'fast']
    assert contract['order'][0] != contract['order'][1]


def test_rehashed_summary_must_match_raw_blocks(regression_manifest, stable_timings, tmp_path):
    out = tmp_path / 'comparison'
    compare(regression_manifest, {'baseline': {}, 'same': {}}, out, rounds=3, repeats=3)
    value = json.loads((out / 'comparison.json').read_text())
    value['rows'][1]['quality']['recall'] = .123
    value.pop('sha256')
    value['sha256'] = canonical_digest(value)
    (out / 'comparison.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='summary differs'):
        verify_comparison(out)


def test_provenance_correction_preserves_measurements_and_checks_binding(regression_manifest, stable_timings, tmp_path):
    from opencv_deploy_proof.comparison import signed
    out = tmp_path / 'comparison'
    original = compare(regression_manifest, {'baseline': {}, 'same': {}}, out, rounds=3, repeats=3)
    contract = json.loads((out / 'comparison-contract.json').read_text())
    correction = {'original_report_sha256': original['sha256'],
                  'original_contract_sha256': contract['sha256'],
                  'manifest_sha256': contract['manifest_sha256'],
                  'corrected_limitations': ['Fresh images on calibration-exposed boards.']}
    target = out / 'provenance-correction.json'
    target.write_text(json.dumps(signed(correction)))
    verified = verify_comparison(out)
    assert verified['original_report'] == original
    assert verified['report']['rows'] == original['rows']
    assert verified['report']['limitations'] == correction['corrected_limitations']
    assert verified['report']['sha256'] != original['sha256']
    target.write_text(json.dumps(signed({**correction, 'original_report_sha256': 'different-study'})))
    with pytest.raises(ValueError, match='not bound'):
        verify_comparison(out)


@pytest.mark.parametrize('candidate,expected', [({'scale': .5}, 2), ({}, 0), ({'scale': .4}, 3)])
def test_ci_exit_statuses_and_report(regression_manifest, stable_timings, tmp_path, monkeypatch, candidate, expected):
    from opencv_deploy_proof.cli import main
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(candidate))
    out = tmp_path / 'ci-result'
    monkeypatch.setattr('sys.argv', ['vision-proof', 'check', '--manifest', str(regression_manifest),
                                   '--candidate-config', str(path), '--out', str(out),
                                   '--rounds', '3', '--repeats', '3'])
    if expected:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == expected
    else:
        main()
    if expected != 3:
        assert (out / 'summary.md').is_file()


def test_comparison_refuses_overwrite_and_bad_protocol(regression_manifest, stable_timings, tmp_path):
    out = tmp_path / 'result'
    compare(regression_manifest, {'baseline': {}, 'same': {}}, out, rounds=3, repeats=3)
    with pytest.raises(ValueError, match='overwrite'):
        compare(regression_manifest, {'baseline': {}, 'same': {}}, out, rounds=3, repeats=3)
    with pytest.raises(ValueError, match='rounds'):
        compare(regression_manifest, {'baseline': {}, 'same': {}}, tmp_path / 'new', rounds=2)


def test_bootstrap_uses_paired_block_ratios():
    result = interval([2, 2, 2, 2])
    assert result['low'] == result['median'] == result['high'] == 2
    with pytest.raises(ValueError):
        interval([1, float('nan'), 2])


def test_invalid_cli_arguments_are_errors_not_regression_codes(monkeypatch):
    from opencv_deploy_proof.cli import main
    monkeypatch.setattr('sys.argv', ['vision-proof', 'check'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 3


def test_duplicate_model_proposal_retains_raw_protocol_evidence(regression_manifest, monkeypatch):
    import sys
    from dataclasses import asdict
    from types import SimpleNamespace
    from opencv_deploy_proof.policy import ProposalError, choose
    from opencv_deploy_proof.evaluate import Requirements
    proposed = {'base_observation': 0, 'parameter': 'threshold', 'value': 128,
                'hypothesis': 'No change', 'falsification': 'Already measured'}
    response = {'output': {'message': {'content': [{'toolUse': {'name': 'propose_experiment', 'input': proposed}}]}},
                'usage': {'inputTokens': 100, 'outputTokens': 20}}
    client = SimpleNamespace(converse=lambda **kwargs: response)
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=lambda name: client))
    baseline = Evaluator(regression_manifest, 'test', 3).run(Config())
    history = [{'result': baseline, 'gate': {'passed': True}}]
    with pytest.raises(ProposalError, match='already measured') as error:
        choose('bedrock', history, asdict(Requirements()), 'test-model')
    assert error.value.evidence['proposal'] == proposed
    assert error.value.evidence['usage']['outputTokens'] == 20
    assert 'toolUse' in error.value.evidence['raw_response']


def test_yolox_decodes_person_boxes_and_applies_confidence_and_nms(tmp_path, monkeypatch):
    from opencv_deploy_proof import pedestrian
    class FakeNet:
        def setInput(self, blob):
            assert blob.shape == (1, 3, 160, 160)

        def forward(self):
            outputs = np.zeros((1, 525, 85), np.float32)
            index = 10 * 20 + 10
            outputs[0, index, :6] = [0, 0, np.log(5), np.log(10), .9, .8]
            return outputs

    monkeypatch.setattr(pedestrian._thread, 'net', FakeNet(), raising=False)
    monkeypatch.setattr(pedestrian._thread, 'model_root', tmp_path, raising=False)
    image = np.zeros((160, 160), np.uint8)
    result = pedestrian.detect_people(image, tmp_path, pedestrian.PedestrianConfig(input_size=160))
    assert result['boxes'] == [[60, 40, 100, 120]]
    with pytest.raises(ValueError):
        pedestrian.PedestrianConfig(input_size=300)
    with pytest.raises(ValueError):
        pedestrian.PedestrianConfig(workers=4, threads=4)


def test_adapter_proposal_validation_uses_its_actual_schema():
    from opencv_deploy_proof.pedestrian import PedestrianConfig
    from opencv_deploy_proof.policy import materialize
    history = [{'result': {'config': PedestrianConfig().to_dict()}}]
    proposed = {'base_observation': 0, 'parameter': 'confidence', 'value': .5,
                'hypothesis': 'Filter low-confidence false positives', 'falsification': 'Recall falls'}
    assert materialize(proposed, history, PedestrianConfig, ('confidence',)).confidence == .5
    with pytest.raises(ValueError):
        materialize({**proposed, 'parameter': 'threshold'}, history, PedestrianConfig, ('confidence',))


def test_reserved_boards_exclude_prior_measured_groups_and_all_prior_images(tmp_path):
    from opencv_deploy_proof.protocol import reserved_manifest
    from opencv_deploy_proof.data import load_manifest
    rows = []
    for name in ['group-old', 'group-reserved']:
        group = tmp_path / 'PCBData' / name
        (group / 'images').mkdir(parents=True)
        (group / 'images_not').mkdir()
        for index in range(3):
            identifier = name + str(index)
            paths = {'image': group / 'images' / (identifier + '_test.jpg'),
                     'template': group / 'images' / (identifier + '_temp.jpg'),
                     'label': group / 'images_not' / (identifier + '.txt')}
            for path in paths.values():
                path.write_text('hash fixture')
            if index == 0:
                rows.append({'id': identifier, 'group': name,
                             'split': 'development' if name == 'group-old' else 'calibration',
                             **{k: str(p.relative_to(tmp_path)) for k, p in paths.items()},
                             'sha256': {k: digest(p) for k, p in paths.items()}})
    value = {'source': 'fixture', 'commit': 'fixture', 'records': rows}
    value['sha256'] = canonical_digest(value)
    original = tmp_path / 'old.json'
    original.write_text(json.dumps(value))
    path = reserved_manifest([original], tmp_path / 'new.json', limit=2)
    data = load_manifest(path)
    assert {r['group'] for r in data['records']} == {'group-reserved'}
    assert 'group-reserved0' not in {r['id'] for r in data['records']}
