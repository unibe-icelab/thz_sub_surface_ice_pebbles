"""Filesystem locations shared by local and Code Ocean executions."""

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("THZ_DATA_ROOT", CODE_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("THZ_RESULTS_ROOT", CODE_ROOT / "results"))
SCRATCH_ROOT = Path(os.environ.get("THZ_SCRATCH_ROOT", CODE_ROOT / ".scratch"))

# Regenerated material-property arrays need a writable location on Code Ocean.
MATERIAL_CACHE_ROOT = Path(
    os.environ.get(
        "THZ_MATERIAL_CACHE_ROOT",
        DATA_ROOT / "material_properties",
    )
)
