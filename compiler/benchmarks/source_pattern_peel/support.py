"""Read-only reuse of frozen P1 utilities and the two source inspectors.

No P1 global, role, plan, preparation, collection or summary is modified.
The small standalone scripts are loaded under distinct names; their existing
absolute `contract` import resolves to the original P1 contract, never P5.
"""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P1_HERE = HERE.parent / "source_pattern"


def load(name, path):
    path = path.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError("benchmark helper module collision: " + name)
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


p1c = load("contract", P1_HERE / "contract.py")
p1 = load("mdslc_p5_frozen_p1_helpers", P1_HERE / "run.py")
legacy = load("mdslc_p5_old_source_inspector", ROOT / "compiler/tests/source_pattern/run.py")
peel = load("mdslc_p5_peeled_source_inspector", ROOT / "compiler/tests/source_pattern_peel/run.py")
