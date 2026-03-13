"""
统一后端入口（语义化命名）：
- 新入口：uvicorn src.backend_api.app:app --reload --port 8000
- 实际实现仍在 src.hyh.app，功能不变
"""

from src.hyh.app import app  # noqa: F401


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.backend_api.app:app", host="127.0.0.1", port=8000, reload=True)
