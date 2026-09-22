from __future__ import annotations

import json
import subprocess
import sys


def test_base_import_does_not_eagerly_load_optional_workflows() -> None:
    command = (
        "import json, sys, pyldt; "
        "print(json.dumps({name: name in sys.modules for name in "
        "['matplotlib', 'astroquery', 'pyldt.reduction']}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "matplotlib": False,
        "astroquery": False,
        "pyldt.reduction": False,
    }
