"""Numerical-runtime guard for the supported Notebook entry's optional fit reuse."""
from pathlib import Path
import sys
import platform
import importlib.metadata
import json
import time
from baseline_data import save_json

def current_runtime():
    return {'python':sys.version,'platform':platform.platform(),
            'versions':{p:importlib.metadata.version(p) for p in
                        ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']}}

def guard_cache_runtime(base,reuse,runtime=None):
    base=Path(base); runtime=current_runtime() if runtime is None else runtime
    path=base/'results/cv/runtime.json'; manifest=base/'results/cv/manifest.json'
    old_runtime=json.loads(path.read_text()) if path.exists() else None
    if reuse and manifest.exists():
        if old_runtime!=runtime:
            raise ValueError('Cached numerical runtime differs or is missing; rerun with REUSE_FITS=False')
    if not reuse and manifest.exists() and old_runtime!=runtime:
        # Preserve prior evidence, but prevent a partially restarted run mixing runtimes.
        archive=base/'results/cv_archives'/str(time.time_ns())
        archive.parent.mkdir(parents=True,exist_ok=True)
        (base/'results/cv').rename(archive)
    save_json(path,runtime)
    return runtime
