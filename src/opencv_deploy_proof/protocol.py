"""Predeclare new image samples without reusing the original experiment images."""

import json
from pathlib import Path

from .data import canonical_digest, digest, load_manifest


def reserved_manifest(originals: list[Path], output: Path, limit=160):
    """Use calibration-pool boards outside prior searches; exclude all prior image IDs."""
    if not originals or type(limit) is not int or limit < 1 or output.exists():
        raise ValueError('need prior manifests, positive limit and a new output path')
    root = originals[0].parent.resolve()
    if output.parent.resolve() != root or any(p.parent.resolve() != root for p in originals):
        raise ValueError('all manifests must remain beside the same dataset')
    previous = [load_manifest(path) for path in originals]
    if len({(v['source'], v['commit']) for v in previous}) != 1:
        raise ValueError('prior manifests must reference the same pinned source')
    excluded_ids = {r['id'] for value in previous for r in value['records']}
    measured_groups = {r['group'] for value in previous for r in value['records'] if r['split'] != 'calibration'}
    groups = [p for p in sorted((root / 'PCBData').glob('group*')) if p.name not in measured_groups]
    if not groups:
        raise ValueError('no calibration-pool source groups outside prior searches remain')
    per_group = []
    for group in groups:
        rows = []
        for image in group.glob('*/*_test.jpg'):
            identifier = image.name.removesuffix('_test.jpg')
            if identifier in excluded_ids:
                continue
            paths = {'image': image, 'template': image.with_name(identifier + '_temp.jpg'),
                     'label': group / (image.parent.name + '_not') / (identifier + '.txt')}
            rows.append({'id': identifier, 'group': group.name, 'split': 'test',
                         **{k: str(p.relative_to(root)) for k, p in paths.items()},
                         'sha256': {k: digest(p) for k, p in paths.items()}})
        per_group.append(sorted(rows, key=lambda r: canonical_digest('visionproof-v3:' + r['id'])))
    records = [rows[i] for i in range(max(map(len, per_group), default=0))
               for rows in per_group if i < len(rows)][:limit]
    if len(records) != limit:
        raise ValueError('insufficient fresh images on reserved groups')
    manifest = {'pipeline': 'pcb-pair', 'source': previous[0]['source'], 'commit': previous[0]['commit'],
                'records': records, 'group_splits': {'test': [p.name for p in groups]},
                'excluded_manifests': [v['sha256'] for v in previous],
                'excluded_measured_groups': sorted(measured_groups),
                'split_method': 'Frozen-config replication on calibration-exposed source-board groups outside prior searches; '
                                'exclude all prior image IDs; hash order balanced round-robin sampling.',
                'limitations': ['These boards were used in earlier baseline calibration; they are not unseen boards.',
                                'Images exclude prior registered samples, but source-board exposure remains.',
                                f'Only {len(groups)} reserved source groups remain; sample counts can be unbalanced.',
                                'Prior calibration images are excluded as well as prior development/test images.']}
    manifest['sha256'] = canonical_digest(manifest)
    output.write_text(json.dumps(manifest, indent=2))
    return output


def fresh_manifest(original: Path, output: Path, limit=64):
    if type(limit) is not int or limit < 1:
        raise ValueError('sample size must be positive')
    if output.exists():
        raise ValueError('fresh manifest must not overwrite a declared protocol')
    if output.parent.resolve() != original.parent.resolve():
        raise ValueError('manifest must remain beside the original dataset')
    previous = load_manifest(original)
    excluded = {row['id'] for row in previous['records']}
    groups = {'development': ['group12100', 'group13000'], 'test': ['group20085', 'group00041']}
    records = []
    for split, names in groups.items():
        per_group = []
        for name in names:
            group = original.parent / 'PCBData' / name
            rows = []
            for image in group.glob('*/*_test.jpg'):
                identifier = image.name.removesuffix('_test.jpg')
                if identifier in excluded:
                    continue
                paths = {'image': image, 'template': image.with_name(identifier + '_temp.jpg'),
                         'label': group / (image.parent.name + '_not') / (identifier + '.txt')}
                rows.append({'id': identifier, 'group': name, 'split': split,
                             **{k: str(p.relative_to(original.parent)) for k, p in paths.items()},
                             'sha256': {k: digest(p) for k, p in paths.items()}})
            per_group.append(sorted(rows, key=lambda row: canonical_digest('visionproof-v2:' + row['id'])))
        selected = [rows[i] for i in range(max(map(len, per_group))) for rows in per_group if i < len(rows)][:limit]
        if len(selected) != limit:
            raise ValueError('insufficient fresh images')
        records.extend(selected)
    manifest = {'source': previous['source'], 'commit': previous['commit'], 'records': records,
                'group_splits': groups, 'excluded_manifest_sha256': previous['sha256'],
                'split_method': 'Predeclared v2, hash-ordered new images; round-robin disjoint development/test board groups.',
                'limitations': ['Images exclude the original experiment, but boards come from its calibration groups.',
                                'This is not a fully unseen board-distribution or official leaderboard evaluation.']}
    manifest['sha256'] = canonical_digest(manifest)
    output.write_text(json.dumps(manifest, indent=2))
    return output
