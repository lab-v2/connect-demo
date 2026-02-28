import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    # When bundled by PyInstaller, files are unpacked to _MEIPASS.
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def main() -> int:
    from streamlit.web import cli as stcli

    base = _base_dir()
    app_file = base / "streamlit_app.py"
    if not app_file.exists():
        print(f"ERROR: streamlit_app.py not found at {app_file}", file=sys.stderr)
        return 1

    os.chdir(base)
    sys.argv = [
        "streamlit",
        "run",
        str(app_file),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        "--server.port=8501",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
