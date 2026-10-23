#!/usr/bin/env python
"""Launcher for tbribbon.generate_configurations (the code lives in src/tbribbon).

    PYTHONPATH=src python scripts/generate/generate_configurations.py --help
"""
import runpy

runpy.run_module("tbribbon.generate_configurations", run_name="__main__", alter_sys=True)
