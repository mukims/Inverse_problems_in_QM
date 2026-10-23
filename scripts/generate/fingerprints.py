#!/usr/bin/env python
"""Launcher for tbribbon.fingerprints (the code lives in src/tbribbon).

    PYTHONPATH=src python scripts/generate/fingerprints.py --help
"""
import runpy

runpy.run_module("tbribbon.fingerprints", run_name="__main__", alter_sys=True)
