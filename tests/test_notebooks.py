"""Notebooks are valid, have no stored outputs, and every code cell compiles (IPython magics stripped)."""
import glob
import os

import nbformat
import pytest

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
NOTEBOOKS = ["phase_1_eval.ipynb", "phase_1_teacher.ipynb", "phase_2_align.ipynb", "phase_3_refine.ipynb"]


def strip_magics(src):
    lines = src.splitlines()
    return "\n".join(l[:len(l) - len(l.lstrip())] + "pass" if l.strip().startswith(("!", "%")) else l for l in lines)


def test_expected_notebooks():
    assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(NB_DIR, "*.ipynb"))) == NOTEBOOKS


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_clean_and_compiles(name):
    nb = nbformat.read(os.path.join(NB_DIR, name), as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code), "notebook has stored outputs"
    assert "REPO_URL" in code[0].source and "DEBUG = True" in code[1].source
    for i, c in enumerate(code):
        compile(strip_magics(c.source), f"{name}[{i}]", "exec")
