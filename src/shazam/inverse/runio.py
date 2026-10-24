"""Run I/O shared by the inverse-design scripts: atomic writes, provenance (code version, file hashes) and the resumable
.jsonl run log.

atomic_write_bytes / atomic_write_json / atomic_save write into <path>.tmp, flush it to disk (fsync) and rename it over
<path>, so a reader or a crash never sees a half-written file (the generator's `*.tmp` sweep relies on that suffix).
code_version() is git HEAD with "-dirty" when tracked files have uncommitted changes, computed once per process (the
first call; the scripts call it when they start). sha256_of(path) is the hex digest of a file.
RunLog is the append-only log of finished targets that makes a long evaluation resumable.
"""
import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path

import paths


def atomic_write_bytes(path, write):
    """Run write(binary file) into path.tmp, flush it to disk and rename it over path."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj, indent=1, allow_nan=False):
    """json.dumps(obj) written atomically (see atomic_write_bytes)."""
    text = json.dumps(obj, indent=indent, allow_nan=allow_nan)
    atomic_write_bytes(path, lambda f: f.write(text.encode()))


def atomic_save(path, save):
    """save(tmp path) (e.g. torch.save or a model's save), then fsync the file and rename it over path."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


@functools.lru_cache(maxsize=None)
def code_version():
    """git HEAD, with "-dirty" appended when tracked files have uncommitted changes (untracked files are ignored);
    "unknown" outside a git checkout. Computed once per process: the first call fixes it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=paths.repo_root(), capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_of(path, chunk=1 << 20):
    """Hex sha256 of the file at path (FileNotFoundError when it does not exist)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class RunLog:
    """Append-only .jsonl of finished targets, one line each: {key_field: key, "cfg": cfg, "record": record} for a target
    that finished, {key_field: key, "cfg": cfg, "error": repr} for one that raised. Reading skips a line a crash cut
    short and refuses lines written with other settings (cfg); a later line for the same key replaces an earlier one, so a
    retried error that now succeeds counts as done.

    done    key -> record of the finished targets
    errors  key -> repr of the error of the targets whose last line is an error
    """

    def __init__(self, path, cfg, key_field="key", tag=None):
        self.path, self.cfg, self.key_field, self.done, self.errors = Path(path), cfg, key_field, {}, {}
        tag = tag or cfg.get("cmd", self.path.stem)
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:               # a line cut short by a crash: that target is simply redone
                    continue
                if rec["cfg"] != cfg:
                    raise SystemExit(f"{self.path} was written with {rec['cfg']}, not {cfg}: delete it to start over")
                key = rec[key_field]
                if "error" in rec:
                    self.done.pop(key, None)
                    self.errors[key] = rec["error"]
                else:
                    self.errors.pop(key, None)
                    self.done[key] = rec["record"]
            print(f"[{tag}] resuming: {len(self.done)} targets already in {self.path.name}"
                  + (f", {len(self.errors)} recorded as failed" if self.errors else ""), flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        if self.path.stat().st_size and not self.path.read_text().endswith("\n"):
            self.fh.write("\n")                            # a crash cut the last line short: new lines start fresh

    def _write(self, line):
        self.fh.write(json.dumps(line) + "\n")
        self.fh.flush()

    def add(self, key, record):
        self._write({self.key_field: key, "cfg": self.cfg, "record": record})
        self.errors.pop(key, None)
        self.done[key] = record

    def add_error(self, key, error):
        """Record that the target `key` raised `error` (an exception or its repr)."""
        text = error if isinstance(error, str) else repr(error)
        self._write({self.key_field: key, "cfg": self.cfg, "error": text})
        self.done.pop(key, None)
        self.errors[key] = text

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
