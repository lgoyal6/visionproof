"""Loopback inspection workbench with verified evidence and bounded local jobs."""

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import parse_qs, urlparse

import cv2

from .data import canonical_digest, load_manifest
from .evaluate import fingerprint, match
from .evidence import cloud_report, export_bundle, public_trace, report, run_summary, verify_run
from .jobs import Jobs
from .pipeline import Config, detect
from .adapters import pipeline_for


def create_server(manifest_path: Path, run: Path, host='127.0.0.1', port=8765,
                  evidence_root: Path | None = None, model: str | None = None):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('workbench must bind to loopback')
    load_manifest(manifest_path)
    verdict = json.loads((run / 'verdict.json').read_text())
    stored = verdict.pop('sha256')
    if canonical_digest(verdict) != stored:
        raise ValueError('verdict digest mismatch')
    if verdict['promotion_status'] != 'preview_eligible':
        raise ValueError('candidate failed independent evaluation; preview deployment refused')
    primary = verify_run(run, manifest_path, require_pass=True)
    current = pipeline_for(manifest_path).fingerprint()
    if primary['selected']['environment']['code_sha256'] != current['code_sha256']:
        raise ValueError('evaluated source changed; rerun qualification before serving')
    if primary['selected']['environment']['opencv_build'] != current['opencv_build']:
        raise ValueError('OpenCV build differs from the evaluated runtime; rerun qualification')
    root = (evidence_root or run.parent).resolve()
    if not run.resolve().is_relative_to(root):
        raise ValueError('primary run must be below evidence root')
    inference_lock, catalog_lock = Lock(), Lock()
    jobs = Jobs(root, manifest_path, model, inference_lock)
    cache = {}
    static = Path(__file__).parent / 'static'

    def catalog():
        items = {}
        with catalog_lock:
            for directory in sorted(root.iterdir()):
                if not directory.is_dir() or not (directory / 'verdict.json').is_file():
                    continue
                try:
                    version = tuple((directory / name).stat().st_mtime_ns for name in
                                    ('verdict.json', 'selection.json', 'contract.json', 'agent-test.json'))
                    if directory.name not in cache or cache[directory.name][0] != version:
                        cache[directory.name] = version, verify_run(directory)
                    items[directory.name] = cache[directory.name][1]
                except (ValueError, KeyError, OSError, TypeError):
                    continue
        return items

    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, kind='application/json', code=200, attachment=None):
            if not isinstance(data, bytes):
                data = (data if isinstance(data, str) else json.dumps(data)).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if attachment:
                self.send_header('Content-Disposition', 'attachment; filename="' + attachment + '"')
            self.end_headers()
            self.wfile.write(data)

        def trusted(self):
            return self.headers.get('Host', '') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.trusted():
                return self.respond({'error': 'unrecognized host'}, code=403)
            try:
                self.route_get()
            except KeyError:
                self.respond({'error': 'unknown run, job or image'}, code=404)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.respond({'error': 'evidence validation failed'}, code=422)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route_get(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path
            if path in ('/', '/app.css', '/app.js'):
                name = 'index.html' if path == '/' else path[1:]
                return self.respond((static / name).read_bytes(), mimetypes.guess_type(name)[0] + '; charset=utf-8')
            if path == '/favicon.ico':
                return self.respond(b'', 'image/x-icon', code=204)
            if path == '/health':
                return self.respond({'status': 'ok', 'opencv': cv2.__version__, 'scope': 'local workbench',
                                     'config': primary['selection']['config']})
            if path == '/api/catalog':
                return self.respond({'default_run': run.name,
                    'runs': sorted([run_summary(item) for item in catalog().values()], key=lambda x: x['created'], reverse=True),
                    'capabilities': {'bedrock': bool(model), 'model': model, 'local_jobs': True},
                    'scope': 'Saved experiments and real local inference. AWS results are recorded benchmarks.'})
            if path == '/api/cloud':
                return self.respond(cloud_report(root))
            if path in ('/api/comparisons', '/api/comparison-export', '/api/comparison-pixels'):
                from .comparison import verify_comparison
                directories = {p.name: p for p in root.iterdir()
                               if p.is_dir() and (p / 'comparison.json').is_file()}
                if path == '/api/comparisons':
                    results = []
                    for identifier, directory in sorted(directories.items()):
                        item = verify_comparison(directory)
                        results.append({'id': identifier, **item['report'],
                                        'protocol': item['report']['limitations']})
                    return self.respond({'comparisons': results})
                directory = directories[query.get('comparison', [''])[0]]
                comparison = verify_comparison(directory)
                if path == '/api/comparison-export':
                    import io
                    import zipfile
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                        for file in directory.iterdir():
                            if file.is_file() and file.suffix in ('.json', '.jsonl', '.md'):
                                archive.writestr(file.name, file.read_bytes())
                        archive.writestr('manifest.json', comparison['manifest_path'].read_bytes())
                    return self.respond(buffer.getvalue(), 'application/zip', attachment='visionproof-' + directory.name + '.zip')
                data = load_manifest(comparison['manifest_path'])
                row = {r['id']: r for r in data['records'] if r['split'] == 'test'}[query.get('id', [''])[0]]
                image = cv2.imread(str(comparison['manifest_path'].parent / row['image']), 0)
                success, encoded = cv2.imencode('.png', image)
                if not success:
                    raise ValueError('image encoding failed')
                return self.respond(encoded.tobytes(), 'image/png')
            if path.startswith('/api/jobs/'):
                return self.respond(jobs.status(path.removeprefix('/api/jobs/')))
            if path not in ('/api/run', '/api/evidence', '/api/trace', '/api/export', '/api/predict', '/api/inspect', '/image', '/api/pixels', '/api/development-pixels'):
                return self.respond({'error': 'not found'}, code=404)
            identifier = query.get('run', [run.name])[0]
            item = catalog()[identifier]
            if path == '/api/run':
                return self.respond(report(item))
            if path == '/api/evidence':
                return self.respond(item['verdict'])
            if path == '/api/trace':
                return self.respond(public_trace(item))
            if path == '/api/export':
                return self.respond(export_bundle(item), 'application/zip', attachment='visionproof-' + identifier + '.zip')
            image_id = query.get('id', [item['selected']['examples'][0]['id']])[0]
            rows = {row['id']: row for row in item['manifest']['records']}
            row = rows[image_id]
            if path == '/api/development-pixels':
                if row['split'] != 'development':
                    raise KeyError('not a development image')
                image = cv2.imread(str(item['manifest_path'].parent / row['image']), 0)
                success, encoded = cv2.imencode('.png', image)
                if not success:
                    raise ValueError('image encoding failed')
                return self.respond(encoded.tobytes(), 'image/png')
            test_rows = {x['id']: x for x in item['selected']['examples']}
            if image_id not in test_rows:
                raise KeyError('unknown test image')
            base = item['manifest_path'].parent
            image = cv2.imread(str(base / row['image']), 0)
            template = cv2.imread(str(base / row['template']), 0)
            if path == '/api/pixels':
                kind = query.get('kind', ['inspection'])[0]
                if kind not in ('inspection', 'template'):
                    raise ValueError('invalid image kind')
                success, encoded = cv2.imencode('.png', template if kind == 'template' else image)
                if not success:
                    raise ValueError('image encoding failed')
                return self.respond(encoded.tobytes(), 'image/png')
            if not item['verdict']['holdout_gate']['passed']:
                return self.respond({'error': 'candidate is not qualified for inference'}, code=409)
            runtime = item['selected']['environment']
            pipeline = pipeline_for(item['manifest_path'])
            runtime_now = pipeline.fingerprint()
            if runtime['code_sha256'] != runtime_now['code_sha256'] or runtime['opencv_build'] != runtime_now['opencv_build']:
                return self.respond({'error': 'recorded runtime differs; requalify before live inference'}, code=409)
            with jobs.lock:
                busy = any(job['status'] == 'running' for job in jobs.jobs.values())
            if busy:
                return self.respond({'error': 'inference paused while qualification measures performance'}, code=409)
            selected_config = pipeline.config_type(**item['selection']['config'])
            with inference_lock:
                cv2.setNumThreads(selected_config.threads)
                started = perf_counter()
                auxiliary = item['manifest_path'].parent / 'models' if pipeline.kind == 'pedestrian-yolox' else template
                answer = pipeline.detect(image, auxiliary, selected_config)
                elapsed = (perf_counter() - started) * 1000
            if path in ('/api/predict', '/api/inspect'):
                labels = test_rows[image_id]['labels']
                return self.respond({'id': image_id, **answer, 'wall_ms': elapsed, 'config': selected_config.to_dict(),
                    'quality': match(answer['boxes'], labels), 'scope': 'Fresh single-image local inference, excludes decoding; not batch or AWS latency.'})
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2 in answer['boxes']:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 1)
            success, encoded = cv2.imencode('.png', overlay)
            if not success:
                raise ValueError('image encoding failed')
            self.respond(encoded.tobytes(), 'image/png')

        def do_POST(self):
            origin = self.headers.get('Origin')
            expected = {'http://127.0.0.1:' + str(self.server.server_port), 'http://localhost:' + str(self.server.server_port)}
            if not self.trusted() or origin not in expected:
                return self.respond({'error': 'same-origin request required'}, code=403)
            if self.path != '/api/jobs':
                return self.respond({'error': 'not found'}, code=404)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond({'error': 'application/json required'}, code=415)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    return self.respond({'error': 'request body must be 1..4096 bytes'}, code=413)
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('expected an object')
                identifier = value.pop('run', run.name)
                if not isinstance(identifier, str) or identifier not in catalog():
                    raise ValueError('unknown qualification run')
                manifest = catalog()[identifier]['manifest_path']
                self.respond(jobs.start(value, manifest), code=202)
            except (ValueError, TypeError) as error:
                self.respond({'error': str(error)}, code=400)
            except RuntimeError as error:
                self.respond({'error': str(error)}, code=409)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.jobs = jobs
    return server


def serve(manifest_path: Path, run: Path, host='127.0.0.1', port=8765,
          evidence_root: Path | None = None, model: str | None = None):
    server = create_server(manifest_path, run, host, port, evidence_root, model)
    print(f'VisionProof listening on http://{host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
