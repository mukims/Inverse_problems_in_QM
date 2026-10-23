#!/usr/bin/env python
"""Launcher for tbribbon.generate_clouds (the code lives in src/tbribbon).

    PYTHONPATH=src python scripts/generate/generate_clouds.py --help
"""
import runpy

runpy.run_module("tbribbon.generate_clouds", run_name="__main__", alter_sys=True)
