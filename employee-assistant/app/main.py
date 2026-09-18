"""Composition root：装配服务与 Router，不写入库或检索业务。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.routers.rag import router
from app.services.assistant import Assistant


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.assistant = (
            assistant if assistant is not None else Assistant(settings=settings)
        )
        yield

    application = FastAPI(
        title="员工助手 · 三层 RAG 后端",
        version="0.2.0",
        lifespan=lifespan,
        description="仅限本地教学：X-Demo-User 是可切换的虚构身份，不是真实认证。默认 demo 不调用 LLM。",
    )
    application.include_router(router)
    return application


app = create_app()
