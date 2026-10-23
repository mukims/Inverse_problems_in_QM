#!/usr/bin/env python
"""Launcher for tbribbon.fingerprints_real (the code lives in src/tbribbon).

    PYTHONPATH=src python scripts/generate/fingerprints_real.py --help
"""
import runpy

runpy.run_module("tbribbon.fingerprints_real", run_name="__main__", alter_sys=True)
