"""One bounded qualification at a time, with persisted progress and no shell execution."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from .experiment import proof
from .journal import append_event, read_events


class Jobs:
    def __init__(self, root: Path, manifest: Path, model: str | None, execution_lock=None):
        self.root, self.manifest, self.model = root, manifest, model
        self.lock, self.jobs = Lock(), {}
        self.execution_lock = execution_lock or Lock()

    def start(self, request: dict, manifest: Path | None = None):
        if set(request) - {'provider', 'budget'}:
            raise ValueError('unknown qualification option')
        provider, budget = request.get('provider', 'rules'), request.get('budget', 4)
        if provider not in ('rules', 'bedrock') or type(budget) is not int or not 1 <= budget <= 4:
            raise ValueError('provider must be rules or bedrock; budget must be an integer from 1 to 4')
        if provider == 'bedrock' and not self.model:
            raise ValueError('Bedrock is not configured for this server')
        with self.lock:
            if any(x['status'] == 'running' for x in self.jobs.values()):
                raise RuntimeError('a qualification is already running')
            identifier = 'local-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
            path = self.root / identifier
            self.jobs[identifier] = {'id': identifier, 'status': 'running', 'provider': provider, 'budget': budget,
                                     'path': path, 'manifest': manifest or self.manifest, 'error': None}
        Thread(target=self._run, args=(identifier,), daemon=True).start()
        return self.status(identifier)

    def _run(self, identifier):
        job = self.jobs[identifier]
        try:
            with self.execution_lock:
                proof(job['manifest'], job['path'], job['provider'], self.model if job['provider'] == 'bedrock' else None,
                      job['budget'], repeats=5)
            with self.lock:
                job['status'] = 'completed'
        except Exception as error:
            message = type(error).__name__
            if isinstance(error, ValueError):
                message += ': ' + str(error)
            # Provider errors can contain signed request details. Keep them out of HTTP responses.
            with self.lock:
                job['status'], job['error'] = 'failed', message
            job['path'].mkdir(parents=True, exist_ok=True)
            append_event(job['path'] / 'events.jsonl', 'failed', {'error': message})
        finally:
            state = {k: v for k, v in job.items() if k not in ('path', 'manifest')}
            (job['path'] / 'job.json').write_text(json.dumps(state, indent=2))

    def status(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError('unknown job')
            job = dict(self.jobs[identifier])
        path = job.pop('path')
        job.pop('manifest')
        job['events'] = read_events(path / 'events.jsonl')
        return job
