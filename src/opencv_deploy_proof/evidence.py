"""Verified evidence views shared by HTTP, exports, and local inference."""

import io
import json
import zipfile
from pathlib import Path

from .data import canonical_digest, load_manifest
from .evaluate import Requirements, gate
from .diagnostics import paired_diagnostics


def read_signed(path: Path):
    value = json.loads(path.read_text())
    signature = value.pop('sha256')
    if canonical_digest(value) != signature:
        raise ValueError(f'{path.name} digest mismatch')
    value['sha256'] = signature
    return value


def verify_run(run: Path, manifest_path: Path | None = None, require_pass=False):
    verdict = read_signed(run / 'verdict.json')
    selection = read_signed(run / 'selection.json')
    contract = json.loads((run / 'contract.json').read_text())
    baseline = read_signed(run / 'baseline-test.json')
    selected = read_signed(run / 'agent-test.json')
    development = read_signed(run / 'baseline-development.json')
    if (verdict['selection'] != selection or selection['contract_sha256'] != canonical_digest(contract)
            or selected['config'] != selection['config']
            or selected['manifest_sha256'] != selection['manifest_sha256']
            or baseline['split'] != 'test' or selected['split'] != 'test'):
        raise ValueError('selection, contract or evaluated configuration mismatch')
    candidates = [development]
    trace = []
    for path in sorted(run.glob('agent-[0-9][0-9].json')):
        entry = json.loads(path.read_text())
        result = entry['result']
        if canonical_digest({k: v for k, v in result.items() if k != 'sha256'}) != result['sha256']:
            raise ValueError('development measurement digest mismatch')
        if result['split'] != 'development' or result['manifest_sha256'] != selection['manifest_sha256']:
            raise ValueError('development measurement split mismatch')
        expected = gate(result, development, Requirements(**contract['requirements']))
        if entry['gate'] != expected:
            raise ValueError('development gate mismatch')
        candidates.append(result)
        trace.append(entry)
    chosen = next((x for x in candidates if x['sha256'] == selection['development_result_sha256']), None)
    if not chosen or chosen['config'] != selection['config']:
        raise ValueError('selection is not bound to a measured development candidate')
    requirements = Requirements(**contract['requirements'])
    if not gate(chosen, development, requirements)['passed']:
        raise ValueError('selection failed its development contract')
    expected = gate(selected, baseline, requirements)
    if verdict['holdout_gate'] != expected:
        raise ValueError('recorded holdout gate mismatch')
    if require_pass and (not expected['passed'] or verdict['promotion_status'] != 'preview_eligible'):
        raise ValueError('candidate failed independent evaluation; preview deployment refused')
    path = manifest_path or Path(contract['manifest_path'])
    manifest = load_manifest(path)
    if manifest['sha256'] != selection['manifest_sha256']:
        raise ValueError('run dataset mismatch')
    return {'run': run, 'manifest_path': path, 'manifest': manifest, 'verdict': verdict,
            'selection': selection, 'contract': contract, 'baseline': baseline,
            'selected': selected, 'development': development, 'trace': trace}


def run_summary(item):
    v = item['verdict']
    return {'id': item['run'].name, 'provider': v['provider'], 'model': v.get('model'),
            'pipeline': item['contract'].get('pipeline', 'pcb-pair'),
            'created': item['contract']['created_utc'], 'status': v['promotion_status'],
            'passed': v['holdout_gate']['passed'], 'speedup': v['speedup_vs_baseline'],
            'sweep_ratio': v['speedup_vs_fixed_sweep'], 'quality': item['selected']['quality'],
            'baseline_quality': item['baseline']['quality'], 'rejected': v['rejected_development_candidates'],
            'test_images': item['selected']['performance']['image_count'],
            'opencv': item['selected']['environment']['opencv'],
            'manifest_sha256': item['manifest']['sha256'], 'sha256': v['sha256']}


def public_trace(item):
    return [{'index': i + 1, 'decision': {k: entry['decision'].get(k) for k in
            ('config', 'hypothesis', 'provider', 'model', 'usage', 'experiment')},
            'gate': entry['gate'], 'quality': entry['result']['quality'],
            'stages_ms': entry['result']['stages_ms'], 'performance': entry['result']['performance'],
            'diagnostics': entry.get('diagnostics') or paired_diagnostics(item['development'], entry['result']),
            'sha256': entry['result']['sha256']}
            for i, entry in enumerate(item['trace'])]


def report(item):
    from .journal import read_events
    original = {row['id']: row for row in item['baseline']['examples']}
    images = [{'id': row['id'], 'shape': row['shape'], 'labels': row['labels'],
               'baseline': original[row['id']], 'selected': row}
              for row in item['selected']['examples']]
    return {**run_summary(item), 'verdict': item['verdict'], 'config': item['selection']['config'],
            'baseline_config': item['baseline']['config'],
            'development_quality': item['development']['quality'],
            'development_performance': item['development']['performance'],
            'baseline_performance': item['baseline']['performance'],
            'selected_performance': item['selected']['performance'],
            'baseline_stages': item['baseline']['stages_ms'],
            'selected_stages': item['selected']['stages_ms'],
            'environment': item['selected']['environment'], 'trace': public_trace(item),
            'images': images, 'events': read_events(item['run'] / 'events.jsonl'),
            'protocol': {k: item['manifest'].get(k) for k in ('source', 'commit', 'split_method', 'limitations')},
            'requirements': item['contract']['requirements']}


def cloud_report(root: Path):
    path = root / 'aws-cool-comparison.json'
    if not path.exists():
        return {'rows': [], 'scope': 'No cloud measurements registered.'}
    value = json.loads(path.read_text())
    for row in value['rows']:
        directory = Path(row['run']).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError('cloud evidence outside registered root')
        selected = read_signed(directory / 'results/selected.json')
        baseline = read_signed(directory / 'results/baseline.json')
        if selected['sha256'] != row['selected_sha256'] or baseline['sha256'] != row['baseline_sha256']:
            raise ValueError('cloud measurement changed')
        if row['gate'] != gate(selected, baseline, Requirements()):
            raise ValueError('cloud gate mismatch')
        row['environment'] = {k: selected['environment'][k] for k in ('opencv', 'numpy', 'machine', 'opencv_module')}
        row.pop('run')
    cleanup = root / 'cool-cleanup-verification.json'
    if cleanup.exists():
        saved = json.loads(cleanup.read_text())
        value['cleanup'] = {'checked_utc': saved['checked_utc'], 'subscription': saved['agreement']['status'],
                            'instances': saved['instances'], 'remaining_worker_volumes': saved['remaining_worker_volumes'],
                            'security_groups': saved['security_groups'], 'key_pairs': saved['key_pairs']}
    value['limitations'] = ['Pip and COOL runs use different OpenCV, NumPy and base-image builds.',
                            'One sequential benchmark per environment is not an isolated COOL effect.',
                            'Cloud runs qualify the earlier rules-selected configuration, not the later Bedrock selection.',
                            'Compute-only estimates exclude startup, idle time, storage, networking, API and software fees.']
    return value


def export_bundle(item):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('qualification.json', json.dumps(report(item), indent=2))
        archive.writestr('selected-config.json', json.dumps(item['selection']['config'], indent=2))
        for path in sorted(item['run'].iterdir()):
            if path.is_file() and path.suffix in ('.json', '.jsonl'):
                archive.writestr('evidence/' + path.name, path.read_bytes())
        archive.writestr('manifest.json', json.dumps(item['manifest'], indent=2))
        source = Path(__file__).parent
        project = source.parents[1]
        for name in ('pyproject.toml', 'uv.lock', 'Dockerfile', 'README.md', 'DESIGN.md', 'NOTICE.md'):
            if (project / name).is_file():
                archive.writestr(name, (project / name).read_bytes())
        for directory in ('configs', 'licenses', '.github/workflows'):
            for path in (project / directory).glob('*'):
                if path.is_file():
                    archive.writestr(str(path.relative_to(project)), path.read_bytes())
        for path in source.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.writestr('src/opencv_deploy_proof/' + str(path.relative_to(source)), path.read_bytes())
        archive.writestr('REPRODUCE.md', '''# VisionProof qualification bundle

This archive contains evidence, source and a frozen configuration. Dataset images are downloaded separately.
Install the repository's locked environment, or install numpy and the exact OpenCV distribution shown in
qualification.json. Source hashes cover data.py, pipeline.py and evaluate.py, plus pedestrian.py for YOLOX. Dataset file hashes and the
custom split are in manifest.json. Put manifest.json beside the downloaded PCBData directory.

PYTHONPATH=src python -m opencv_deploy_proof.cli evaluate --manifest /path/to/data/manifest.json --split test --config selected-config.json --repeats 15 --out result.json

The bundle is finite benchmark evidence, not a guarantee of factory performance or future accuracy.
The model proposes configurations; independent scoring gates promotion. Historical search failures remain
available in the workbench. Hashes detect accidental changes, not a malicious party who rewrites all hashes.
''')
    return data.getvalue()
