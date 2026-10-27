"""The notebook is valid, has no stored outputs, and every code cell compiles (IPython magics stripped)."""
import glob
import os

import nbformat

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")


def strip_magics(src):
    lines = src.splitlines()
    return "\n".join(l[:len(l) - len(l.lstrip())] + "pass" if l.strip().startswith(("!", "%")) else l for l in lines)


def test_only_phase_1_notebook():
    assert [os.path.basename(p) for p in glob.glob(os.path.join(NB_DIR, "*.ipynb"))] == ["phase_1_teacher.ipynb"]


def test_notebook_clean_and_compiles():
    path = os.path.join(NB_DIR, "phase_1_teacher.ipynb")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code), "notebook has stored outputs"
    assert "REPO_URL" in code[0].source and "DEBUG = True" in code[1].source
    for i, c in enumerate(code):
        compile(strip_magics(c.source), f"phase_1_teacher.ipynb[{i}]", "exec")
