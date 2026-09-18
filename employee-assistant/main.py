"""兼容原启动命令 uvicorn main:app；推荐 uvicorn app.main:app。"""

from app.main import app  # noqa: F401
