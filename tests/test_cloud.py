import subprocess

import pytest

pytest.importorskip("boto3")

from scripts import cloud_run


def test_bootstrap_probe_recovers_after_a_transient_ssh_timeout(monkeypatch):
    attempts = iter([subprocess.TimeoutExpired(["ssh"], 10),
                     subprocess.CompletedProcess(["ssh"], 0)])

    def probe(*args, **kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cloud_run.subprocess, "run", probe)
    assert cloud_run.ssh_ready(["ssh", "ubuntu@example.invalid"]) is False
    assert cloud_run.ssh_ready(["ssh", "ubuntu@example.invalid"]) is True
