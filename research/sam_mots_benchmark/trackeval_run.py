"""Run an official TrackEval script unchanged, with Python's csv field-size limit raised.

TrackEval reads MOTS files with the csv module, whose default field limit (131,072 chars)
rejects valid RLE strings of very fragmented masks (SAM 3.1, MOTS20-11, track 2027, frames
468-485: up to ~3,000 fragments). Nothing else changes.

    python trackeval_run.py scripts/run_mots_challenge.py --USE_PARALLEL False ...
"""
import csv
import runpy
import sys

csv.field_size_limit(sys.maxsize)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
