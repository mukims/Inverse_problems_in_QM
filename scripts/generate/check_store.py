#!/usr/bin/env python
"""Launcher for tbribbon.check_store (the code lives in src/tbribbon).

    PYTHONPATH=src python scripts/generate/check_store.py --help
"""
import runpy

runpy.run_module("tbribbon.check_store", run_name="__main__", alter_sys=True)
