"""Notebooks are valid, have no stored outputs, and every code cell compiles (IPython magics stripped).
Full execution in TINY mode: .venv/bin/python scripts/run_notebooks_tiny.py (slower, not part of pytest)."""
import glob
import os

import nbformat
import pytest

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
NOTEBOOKS = sorted(glob.glob(os.path.join(NB_DIR, "0*.ipynb")))


def strip_magics(src):
    lines = src.splitlines()
    return "\n".join(l[:len(l) - len(l.lstrip())] + "pass" if l.strip().startswith(("!", "%")) else l for l in lines)


def test_three_notebooks_exist():
    assert [os.path.basename(p) for p in NOTEBOOKS] == ["00_baseline.ipynb", "01_align_only.ipynb", "02_cr_only.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=os.path.basename)
def test_notebook_clean_and_compiles(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code), "notebook has stored outputs"
    assert "REPO_URL" in code[0].source and "DEBUG = True" in code[1].source
    for i, c in enumerate(code):
        compile(strip_magics(c.source), f"{os.path.basename(path)}[{i}]", "exec")
