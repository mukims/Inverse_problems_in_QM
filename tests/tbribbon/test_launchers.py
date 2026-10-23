import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LAUNCHERS = ["generate_clouds", "check_store", "generate_configurations", "fingerprints", "fingerprints_real"]


def _run(args, **kw):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, *args], env=env, cwd=REPO, capture_output=True, text=True, **kw)


@pytest.mark.parametrize("name", LAUNCHERS)
def test_every_generation_launcher_shows_its_help(name):
    run = _run([str(REPO / "scripts" / "generate" / f"{name}.py"), "--help"])
    assert run.returncode == 0 and "usage" in run.stdout.lower(), run.stderr


def test_generate_clouds_launcher_writes_a_cloud_with_worker_processes(tmp_path):
    run = _run([str(REPO / "scripts" / "generate" / "generate_clouds.py"),
                "--store", str(tmp_path / "s"), "--grid", "models", "--models", "triangular/zigzag/N4",
                "--densities", "0.02", "--spec-version", "v3", "--formula", "caroli", "--n-seeds", "2", "--n-jobs", "2"])
    assert run.returncode == 0, run.stderr[-800:]
    meta = json.loads((tmp_path / "s" / "triangular" / "zigzag" / "N4" / "meta.json").read_text())
    assert len(meta["clouds"]) == 1
