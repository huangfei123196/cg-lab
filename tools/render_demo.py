"""Record the real Taichi GUI output produced by src.Work0.main."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "demo.gif"


def main() -> None:
    env = os.environ.copy()
    env["CG_LAB_RECORD_GIF"] = str(OUTPUT)
    env.setdefault("CG_LAB_NUM_PARTICLES", "10000")

    subprocess.run(
        [sys.executable, "-m", "src.Work0.main"],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
