"""Predeclared, randomized block measurements for CI and replication."""

import json
import random
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .adapters import pipeline_for
from .data import canonical_digest, load_manifest
from .evaluate import Requirements, gate, metrics
from .journal import append_event


def signed(value):
    return {**value, 'sha256': canonical_digest(value)}


def interval(values, seed=1729):
    """Paired blocks are the resampling unit, not individual timing iterations."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('need at least three finite positive paired ratios')
    samples = rng.choice(values, size=(10000, len(values)), replace=True)
    low, high = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    return {'median': float(np.median(values)), 'low': float(low), 'high': float(high),
            'confidence': .95, 'method': 'percentile bootstrap of median paired block ratios, 10000 resamples'}


def summarize(contract, blocks):
    names = list(contract['configs'])
    if len(blocks) != contract['rounds']:
        raise ValueError('incomplete comparison')
    baseline_name = 'baseline'
    rows = []
    for name in names:
        measurements = [block['results'][name] for block in blocks]
        # The detector must be deterministic on exactly the registered examples.
        first = measurements[0]
        stable = [{k: result[k] for k in ('quality', 'examples', 'config', 'manifest_sha256', 'split')}
                  for result in measurements]
        if any(value != stable[0] for value in stable[1:]):
            raise ValueError('non-deterministic predictions or mismatched comparison inputs')
        representative = {**first, 'performance': {**first['performance'],
                            'median_batch_ms': statistics.median(m['performance']['median_batch_ms'] for m in measurements),
                            'p95_batch_ms': statistics.median(m['performance']['p95_batch_ms'] for m in measurements)}}
        baseline = blocks[0]['results'][baseline_name]
        baseline = {**baseline, 'performance': {**baseline['performance'],
                    'p95_batch_ms': statistics.median(b['results'][baseline_name]['performance']['p95_batch_ms'] for b in blocks)}}
        paired = [b['results'][baseline_name]['performance']['median_batch_ms'] /
                  b['results'][name]['performance']['median_batch_ms'] for b in blocks]
        groups = {}
        membership = contract['groups']
        for example in first['examples']:
            counts = groups.setdefault(membership[example['id']], {'tp': 0, 'fp': 0, 'fn': 0})
            for key in counts:
                counts[key] += example[key]
        baseline_examples = {e['id']: e for e in blocks[0]['results'][baseline_name]['examples']}
        lost, recovered = [], []
        for example in first['examples']:
            before = set(baseline_examples[example['id']]['missed_indices'])
            after = set(example['missed_indices'])
            lost.extend({'id': example['id'], 'annotation': i, 'box': example['labels'][i][:4]}
                        for i in sorted(after - before))
            recovered.extend({'id': example['id'], 'annotation': i} for i in sorted(before - after))
        rows.append({'name': name, 'config': first['config'], 'quality': first['quality'],
                     'median_batch_ms': representative['performance']['median_batch_ms'],
                     'median_round_p95_ms': representative['performance']['p95_batch_ms'],
                     'paired_speedup': interval(paired, contract['seed']),
                     'gate': gate(representative, baseline, Requirements(**contract['requirements'])),
                     'round_gates': [gate(b['results'][name], b['results'][baseline_name],
                                          Requirements(**contract['requirements'])) for b in blocks],
                     'by_group': {key: metrics(**counts) for key, counts in groups.items()},
                     'lost_count': len(lost), 'recovered_count': len(recovered), 'lost_examples': lost[:12],
                     'examples': first['examples'], 'environment': first['environment']})
    pairwise = []
    for left in names:
        for right in names:
            if names.index(left) >= names.index(right):
                continue
            ratios = [b['results'][right]['performance']['median_batch_ms'] /
                      b['results'][left]['performance']['median_batch_ms'] for b in blocks]
            pairwise.append({'numerator': right, 'denominator': left,
                             'speedup_of_denominator': interval(ratios, contract['seed'])})
    return signed({'kind': 'randomized-comparison', 'created_utc': contract['created_utc'],
                   'pipeline': contract['pipeline'], 'contract_sha256': contract['sha256'],
                   'manifest_sha256': contract['manifest_sha256'], 'rounds': contract['rounds'],
                   'repeats': contract['repeats'], 'seed': contract['seed'],
                   'image_count': len(contract['groups']), 'group_count': len(set(contract['groups'].values())),
                   'rows': rows, 'pairwise': pairwise,
                   'limitations': contract['protocol'] + [
                       'Configurations and measurement order were frozen before evaluation; no new search is performed.',
                       'Timing intervals resample local measurement blocks, not independent machines or datasets.',
                       'Aggregate p95 gate uses the median of per-round warm batch p95 values, not request p95.',
                       'Per-group quality is descriptive. Few source groups limit generalization.',
                       'This does not establish LLM search superiority or an isolated COOL speed effect.']})


def markdown_report(report):
    lines = ['# VisionProof regression gate', '',
             f"Pipeline: {report['pipeline']}. {report['image_count']} images, {report['rounds']} randomized blocks.", '',
             '| Configuration | Gate | Recall | Precision | Batch ms | Paired speedup (95% interval) |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in report['rows']:
        speed = row['paired_speedup']
        lines.append(f"| {row['name']} | {'PASS' if row['gate']['passed'] else 'BLOCKED'} | "
                     f"{row['quality']['recall']:.3f} | {row['quality']['precision']:.3f} | "
                     f"{row['median_batch_ms']:.1f} | {speed['median']:.2f}x [{speed['low']:.2f}, {speed['high']:.2f}] |")
    lines += ['', 'The gate checks recall, precision and median round p95 against the frozen baseline.',
              'Exit 0: every candidate passed. Exit 2: a measured regression. Exit 3: invalid input or execution error.', '',
              '## Scope', '', *['- ' + item for item in report['limitations']], '']
    return '\n'.join(lines)


def compare(manifest: Path, configs: dict, out: Path, rounds=5, repeats=5, seed=1729, provenance=None):
    if (type(rounds) is not int or not 3 <= rounds <= 30 or type(repeats) is not int
            or not 3 <= repeats <= 31 or type(seed) is not int):
        raise ValueError('rounds must be 3..30, repeats 3..31, seed an integer')
    if not 2 <= len(configs) <= 6 or 'baseline' not in configs:
        raise ValueError('comparison requires baseline and one to five named candidates')
    if any(not name or not name.replace('_', '').replace('-', '').isalnum() for name in configs):
        raise ValueError('configuration names must be simple identifiers')
    pipeline = pipeline_for(manifest)
    configurations = {name: pipeline.config_type(**config) for name, config in configs.items()}
    data = load_manifest(manifest)
    test_rows = [r for r in data['records'] if r['split'] == 'test']
    if not test_rows:
        raise ValueError('comparison requires test images')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('comparison output must be empty; never overwrite evidence')
    rng = random.Random(seed)
    # Shuffled rotations balance positions within each complete cycle.
    order = []
    while len(order) < rounds:
        names = list(configs)
        rng.shuffle(names)
        rotations = [names[i:] + names[:i] for i in range(len(names))]
        rng.shuffle(rotations)
        order.extend(rotations)
    contract = signed({'created_utc': datetime.now(timezone.utc).isoformat(), 'pipeline': pipeline.kind,
                       'manifest_path': str(manifest.resolve()), 'manifest_sha256': data['sha256'],
                       'configs': {k: v.to_dict() for k, v in configurations.items()},
                       'groups': {r['id']: r['group'] for r in test_rows},
                       'rounds': rounds, 'repeats': repeats, 'seed': seed, 'order': order[:rounds],
                       'requirements': asdict(Requirements()), 'provenance': provenance or {},
                       'protocol': data.get('limitations', [])})
    (out / 'comparison-contract.json').write_text(json.dumps(contract, indent=2))
    journal = out / 'comparison-events.jsonl'
    append_event(journal, 'comparison_frozen', {'contract_sha256': contract['sha256']})
    evaluator = pipeline.evaluator(manifest, 'test', repeats)
    blocks = []
    for index, names in enumerate(contract['order']):
        block = {'index': index + 1, 'order': names, 'results': {}}
        for name in names:
            append_event(journal, 'measuring', {'block': index + 1, 'name': name})
            result = evaluator.run(configurations[name])
            if result['manifest_sha256'] != data['sha256']:
                raise ValueError('manifest changed during comparison')
            block['results'][name] = result
            print(json.dumps({'block': index + 1, 'name': name,
                              'median_batch_ms': result['performance']['median_batch_ms'],
                              'quality': result['quality']}), flush=True)
        block = signed(block)
        (out / f'block-{index + 1:02d}.json').write_text(json.dumps(block, indent=2))
        blocks.append(block)
    report = summarize(contract, blocks)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    (out / 'summary.md').write_text(markdown_report(report))
    append_event(journal, 'comparison_completed', {'report_sha256': report['sha256']})
    return report


def verify_comparison(directory: Path):
    from .evidence import read_signed
    from .journal import read_events
    contract = read_signed(directory / 'comparison-contract.json')
    report = read_signed(directory / 'comparison.json')
    manifest = Path(contract['manifest_path'])
    data = load_manifest(manifest)
    if data['sha256'] != contract['manifest_sha256']:
        raise ValueError('comparison dataset mismatch')
    expected_groups = {r['id']: r['group'] for r in data['records'] if r['split'] == 'test'}
    if contract['groups'] != expected_groups or contract['pipeline'] != pipeline_for(manifest).kind:
        raise ValueError('comparison source-group or pipeline binding mismatch')
    blocks = [read_signed(directory / f'block-{index + 1:02d}.json') for index in range(contract['rounds'])]
    for index, block in enumerate(blocks):
        if block['index'] != index + 1 or block['order'] != contract['order'][index]:
            raise ValueError('comparison measurement order mismatch')
        if set(block['results']) != set(contract['configs']):
            raise ValueError('comparison configuration set mismatch')
        for name, result in block['results'].items():
            if (result['config'] != contract['configs'][name] or result['split'] != 'test'
                    or result['manifest_sha256'] != contract['manifest_sha256']
                    or canonical_digest({k: v for k, v in result.items() if k != 'sha256'}) != result['sha256']
                    or {e['id'] for e in result['examples']} != set(contract['groups'])):
                raise ValueError('comparison measurement binding mismatch')
    if summarize(contract, blocks) != report:
        raise ValueError('comparison summary differs from measured evidence')
    events = read_events(directory / 'comparison-events.jsonl')
    if (not events or events[0]['kind'] != 'comparison_frozen'
            or events[0]['payload']['contract_sha256'] != contract['sha256']
            or events[-1]['payload'].get('report_sha256') != report['sha256']):
        raise ValueError('comparison event binding mismatch')
    original_report = report
    correction_path = directory / 'provenance-correction.json'
    if correction_path.exists():
        correction = read_signed(correction_path)
        if (correction['original_report_sha256'] != report['sha256']
                or correction['original_contract_sha256'] != contract['sha256']
                or correction['manifest_sha256'] != contract['manifest_sha256']):
            raise ValueError('provenance correction is not bound to this study')
        if (not isinstance(correction.get('corrected_limitations'), list)
                or not correction['corrected_limitations']
                or not all(isinstance(x, str) and x for x in correction['corrected_limitations'])):
            raise ValueError('provenance correction requires explicit limitations')
        report = signed({**{k: v for k, v in report.items() if k != 'sha256'},
                         'limitations': correction['corrected_limitations'],
                         'original_report_sha256': original_report['sha256'],
                         'provenance_correction': correction})
    return {'directory': directory, 'contract': contract, 'report': report,
            'original_report': original_report, 'manifest_path': manifest}
