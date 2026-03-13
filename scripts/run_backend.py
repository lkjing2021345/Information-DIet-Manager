from pathlib import Path
import sys
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    uvicorn.run("src.backend_api.app:app", host="127.0.0.1", port=8000, reload=True)
