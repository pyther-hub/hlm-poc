"""Execute 00 -> 01 -> 02 -> 00 end to end in TINY mode (tiny random models, fake rows, CPU, 2 gloo processes).
Outputs are not written back; the notebooks stay clean. Needs: uv pip install nbclient ipykernel
    .venv/bin/python scripts/run_notebooks_tiny.py"""
import os
import shutil
import sys

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")


def run(name):
    nb = nbformat.read(os.path.join(NB_DIR, name), as_version=4)
    panel = next(c for c in nb.cells if c.cell_type == "code" and "TINY = False" in c.source)
    panel.source = panel.source.replace("TINY = False", "TINY = True")
    print(f"--- {name}", flush=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": NB_DIR}}).execute()
    for cell in nb.cells:  # kernel-side prints land in cell outputs (worker prints go straight to this terminal)
        for out in cell.get("outputs", []):
            sys.stdout.write(out.get("text", ""))


if __name__ == "__main__":
    for d in ("cache/tiny", "runs/tiny"):
        shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)
    for name in ("00_baseline.ipynb", "01_align_only.ipynb", "02_cr_only.ipynb", "00_baseline.ipynb"):
        run(name)
