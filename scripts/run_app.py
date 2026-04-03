from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    app_file = root / "app" / "main.py"

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    subprocess.run(cmd, check=True, cwd=str(root))


if __name__ == "__main__":
    main()
