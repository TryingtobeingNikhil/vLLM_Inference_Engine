"""
Root conftest.py — makes `inference_engine` importable during pytest runs
regardless of which directory pytest is invoked from.

This file must sit at the PageServe/ root (one level above inference_engine/).
"""
import sys
import os

# Insert the repo root into sys.path so that `import inference_engine.*` works.
sys.path.insert(0, os.path.dirname(__file__))
