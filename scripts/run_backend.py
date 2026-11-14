from pathlib import Path
import argparse
import sys
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the single-process local IDM API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Development only: reload source changes")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    uvicorn.run("src.backend_api.app:app", host="127.0.0.1", port=args.port, reload=args.reload, proxy_headers=False)
