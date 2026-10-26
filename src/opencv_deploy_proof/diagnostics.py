"""Paired development-only error attribution for configuration experiments."""

from collections import Counter


def paired_diagnostics(baseline: dict, candidate: dict) -> dict:
    if baseline['manifest_sha256'] != candidate['manifest_sha256'] or baseline['split'] != candidate['split']:
        raise ValueError('diagnostics require identical split and manifest')
    original = {row['id']: row for row in baseline['examples']}
    lost, recovered, buckets = [], [], {}
    for row in candidate['examples']:
        before = original[row['id']]
        if before['labels'] != row['labels']:
            raise ValueError('annotation mismatch')
        old, new = set(before['missed_indices']), set(row['missed_indices'])
        for index, label in enumerate(row['labels']):
            width, height = label[2] - label[0], label[3] - label[1]
            bucket = 'small' if min(width, height) < 20 else 'medium' if min(width, height) < 40 else 'large'
            counts = buckets.setdefault(bucket, Counter())
            counts['total'] += 1
            counts['baseline_missed'] += index in old
            counts['candidate_missed'] += index in new
            if index in new - old:
                lost.append({'image_id': row['id'], 'annotation_index': index,
                             'box': label[:4], 'width': width, 'height': height, 'size': bucket})
            if index in old - new:
                recovered.append({'image_id': row['id'], 'annotation_index': index, 'box': label[:4]})
    changed = {key: {'before': baseline['config'][key], 'after': value}
               for key, value in candidate['config'].items() if value != baseline['config'][key]}
    stages = baseline['stages_ms']
    total = sum(stages.values()) or 1
    return {'changed_parameters': changed, 'newly_missed_count': len(lost),
            'recovered_count': len(recovered), 'newly_missed_examples': lost[:8],
            'size_buckets': {key: dict(value) for key, value in buckets.items()},
            'baseline_stage_share': {key: value / total for key, value in stages.items()},
            'interpretation': 'Paired matching deltas are observations, not proof of causation. '
                              'Changing multiple parameters confounds attribution. Concurrency cannot restore lost pixels.'}
