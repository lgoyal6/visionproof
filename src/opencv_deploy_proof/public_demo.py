"""Export a verified, read-only replay without deploying execution endpoints."""

import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote

import cv2

from .comparison import verify_comparison
from .data import canonical_digest, load_manifest
from .evidence import cloud_report, report, run_summary, verify_run


def sanitize(value):
    """Public views are projections; retained hashes reference original evidence."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()
                if k not in {'manifest_path', 'dataset_path', 'raw_response', 'prompt',
                             'instance_id', 'account_id', 'arn', 'key_name', 'security_group_id'}}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'/(?:Users|home|tmp|private|opt|usr)/[^\s"\']+', '[local path]', value)
        value = re.sub(r'\b(?:i|sg|vol)-[0-9a-f]{8,}\b', '[resource]', value)
        value = re.sub(r'\b\d{12}\b', '[account]', value)
        return value
    return value


def export_public(root: Path, out: Path, runs: list[str], default_run: str):
    if default_run not in runs or not runs:
        raise ValueError('default run must be explicitly included')
    if out.exists() and any(out.iterdir()):
        raise ValueError('public output must be empty')
    root = root.resolve()
    if any(Path(name).name != name or name in ('.', '..') for name in runs):
        raise ValueError('run names must be simple directory names')
    items = [verify_run(root / name) for name in runs]
    studies = [verify_comparison(p) for p in sorted(root.iterdir())
               if p.is_dir() and (p / 'comparison.json').exists()]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir()
    routes = {}

    def write(route, value, extension='json'):
        target = 'data/' + canonical_digest(route)[:24] + '.' + extension
        payload = json.dumps(sanitize(value), indent=2).encode() if extension == 'json' else value
        (out / target).write_bytes(payload)
        routes[route] = '/' + target

    def pixels(route, manifest, row, kind='image'):
        image = cv2.imread(str(manifest.parent / row[kind]), 0)
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('could not encode sample')
        write(route, encoded.tobytes(), 'png')

    for item in items:
        identifier = quote(item['run'].name)
        view = report(item)
        view['images'] = view['images'][:3]
        rows = {r['id']: r for r in item['manifest']['records']}
        for example in view['images']:
            row = rows[example['id']]
            for kind in ('inspection', 'template'):
                pixels(f'/api/pixels?run={identifier}&id={quote(row["id"])}&kind={kind}',
                       item['manifest_path'], row, 'template' if kind == 'template' else 'image')
        for entry in view['trace']:
            diagnostic = entry['diagnostics']
            samples = diagnostic.get('newly_missed_examples', [])
            diagnostic['newly_missed_examples'] = samples[:1]
            for sample in samples[:1]:
                row = rows[sample['image_id']]
                pixels(f'/api/development-pixels?run={identifier}&id={quote(row["id"])}', item['manifest_path'], row)
        write(f'/api/run?run={identifier}', view)
        write(f'/api/evidence?run={identifier}', item['verdict'])
        write(f'/api/export?run={identifier}', {'scope': 'Sanitized replay summary. Reproduce with the source repository.',
                                              'qualification': view, 'selected_config': item['selection']['config']})
    write('/api/catalog', {'default_run': default_run, 'runs': [run_summary(i) for i in items],
                          'capabilities': {'bedrock': False, 'local_jobs': False},
                          'scope': 'Read-only public replay; no execution endpoints.'})
    comparisons = []
    for study in studies:
        identifier = study['directory'].name
        value = {'id': identifier, **study['report'], 'protocol': study['report']['limitations']}
        rows = {r['id']: r for r in load_manifest(study['manifest_path'])['records']}
        for candidate in value['rows']:
            samples = [candidate['examples'][0]]
            if candidate['lost_examples']:
                lost_id = candidate['lost_examples'][0]['id']
                samples += [e for e in candidate['examples'] if e['id'] == lost_id]
            candidate['examples'] = samples
            for example in samples:
                row = rows[example['id']]
                pixels(f'/api/comparison-pixels?comparison={quote(identifier)}&id={quote(row["id"])}',
                       study['manifest_path'], row)
        comparisons.append(value)
        write('/api/comparison-export?comparison=' + quote(identifier), value)
    write('/api/comparisons', {'comparisons': comparisons})
    write('/api/cloud', cloud_report(root))
    static = Path(__file__).parent / 'static'
    for name in ('app.css', 'app.js', 'about.html'):
        shutil.copy2(static / name, out / name)
    html = (static / 'index.html').read_text().replace('<script defer src="/app.js">',
        '<script src="/public-config.js"></script>\n    <script defer src="/app.js">')
    (out / 'index.html').write_text(html)
    (out / 'public-config.js').write_text('window.VISIONPROOF_PUBLIC = ' + json.dumps({'routes': routes}) + ';\n')
    (out / 'vercel.json').write_text(json.dumps({'framework': None, 'buildCommand': None,
                                               'outputDirectory': '.', 'cleanUrls': False}))
    return {'runs': len(items), 'studies': len(studies), 'routes': len(routes), 'directory': str(out)}
