"""Convert Task 2/3 scripts into Colab-friendly notebooks.

Run from the handout folder:

    python make_colab_notebooks.py

The generated notebooks keep the Python files as source of truth. In Colab,
upload this handout folder or a zip containing:

    data/
    hw2_utils.py
    task1_data.py
    task2_pointnet.py
    task3_transformer.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown_cell(text: str) -> dict:
    text = textwrap.dedent(text).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    text = textwrap.dedent(text).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


def split_script(source: str) -> tuple[str, str]:
    marker = 'if __name__ == "__main__":'
    if marker not in source:
        return source, ""
    before, after = source.split(marker, 1)
    return before.rstrip() + "\n", marker + after


def notebook_for(script_name: str, title: str, extra_notes: str, extra_required: list[str] | None = None) -> dict:
    source = (ROOT / script_name).read_text(encoding="utf-8")
    definitions, script_entrypoint = split_script(source)
    required = ["data", "hw2_utils.py", "task1_data.py", script_name]
    if extra_required:
        required.extend(extra_required)

    cells = [
        markdown_cell(
            f"""
            # {title}

            This notebook was generated from `{script_name}` by
            `make_colab_notebooks.py`.

            Before running, make sure the Colab working directory contains the
            handout files:

            - `data/`
            - `hw2_utils.py`
            - `task1_data.py`
            - `task2_pointnet.py`
            - `task3_transformer.py`

            {extra_notes}
            """
        ),
        code_cell(
            """
            # Optional: upload a zip of the handout folder if the files are not already here.
            # from google.colab import files
            # uploaded = files.upload()
            # !unzip -o handout.zip
            # %cd handout
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import os

            # The original script uses Path(__file__).resolve().parent.
            # Notebooks do not define __file__, so provide the same meaning here.
            __file__ = str(Path.cwd() / "{script_name}")

            required = {required!r}
            missing = [path for path in required if not Path(path).exists()]
            if missing:
                raise FileNotFoundError(f"Missing files/directories in Colab working directory: {{missing}}")

            print("Working directory:", Path.cwd())
            print("CUDA available:", __import__("torch").cuda.is_available())
            """
        ),
        markdown_cell("## Script Definitions"),
        code_cell(definitions),
        markdown_cell("## Train"),
        code_cell(
            """
            # This runs the same main() function as the Python script.
            import sys
            sys.argv = [__file__]
            metrics = main()
            metrics
            """
        ),
        markdown_cell("## Final Test"),
        code_cell(
            f"""
            # Only run this once after validation choices are frozen.
            # import sys
            # from unittest.mock import patch
            # with patch.object(sys, "argv", ["{script_name}", "--test"]):
            #     test_metrics = main()
            # test_metrics
            """
        ),
    ]
    if script_entrypoint:
        cells.append(markdown_cell("## Original Entrypoint"))
        cells.append(markdown_cell("For reference only (do not run again):\n\n```python\n" + script_entrypoint + "\n```"))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(
    script_name: str,
    output_name: str,
    title: str,
    extra_notes: str,
    extra_required: list[str] | None = None,
) -> None:
    notebook = notebook_for(script_name, title, extra_notes, extra_required)
    output_path = ROOT / output_name
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.name}")


def main() -> None:
    write_notebook(
        "task2_pointnet.py",
        "task2_pointnet_colab.ipynb",
        "Task 2: PointNet Segmenter",
        "Task 2 saves `outputs/task2/pointnet.pt`, which Task 3 can use for the comparison figure.",
    )
    write_notebook(
        "task3_transformer.py",
        "task3_transformer_colab.ipynb",
        "Task 3: Transformer Segmenter",
        "Run Task 2 first: Task 3 requires its PointNet checkpoint for `model_comparison.png`.",
        extra_required=["task2_pointnet.py"],
    )


if __name__ == "__main__":
    main()
