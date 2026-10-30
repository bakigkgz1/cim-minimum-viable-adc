# Rebuilds Figs. 1-5 and Tables 2-4 / S3-S4 of the manuscript from results/full_rerun_v1.
# Usage (repo root):  python analysis/unit_column/run_all.py   [P3_RESULTS=... P3_OUT=...]
import os, runpy
here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(here, 'figures.py'), run_name='__main__')   # also writes hw.json
runpy.run_path(os.path.join(here, 'tables.py'), run_name='__main__')    # writes tables.json
