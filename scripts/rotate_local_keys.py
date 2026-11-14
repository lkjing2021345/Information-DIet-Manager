"""Rotate both capability keys while the local API is stopped; never print keys."""
import argparse
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    # Help and argument errors must not load DB configuration or touch key files.
    from src.hyh.db import DB_PATH
    from src.hyh.security import credential_path, process_ownership, write_new_credentials

    if "IDM_ADMIN_TOKEN" in os.environ or "IDM_COLLECTOR_TOKEN" in os.environ:
        raise RuntimeError("Environment keys are configured; rotate both environment values instead.")
    with process_ownership(DB_PATH):
        path = credential_path(DB_PATH)
        write_new_credentials(path, replace=True)
    print(f"Keys rotated: {path}. Restart the service and reconnect both clients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
