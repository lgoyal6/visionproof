"""Append-only local event journal with a verifiable hash chain."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .data import canonical_digest


def append_event(path: Path, kind: str, payload: dict):
    events = read_events(path)
    event = {'sequence': len(events), 'time': datetime.now(timezone.utc).isoformat(),
             'kind': kind, 'payload': payload,
             'previous_sha256': events[-1]['sha256'] if events else None}
    event['sha256'] = canonical_digest(event)
    with path.open('a') as stream:
        stream.write(json.dumps(event) + '\n')
        stream.flush()
    return event


def read_events(path: Path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        signature = event.pop('sha256')
        if (canonical_digest(event) != signature or event['sequence'] != len(events)
                or event['previous_sha256'] != (events[-1]['sha256'] if events else None)):
            raise ValueError('event journal verification failed')
        event['sha256'] = signature
        events.append(event)
    return events
