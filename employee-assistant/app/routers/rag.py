"""Router：只处理 HTTP 输入、依赖注入与异常到状态码的转换。"""

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.errors import IngestForbiddenError, StorageError, UnknownUserError
from app.models import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.services.assistant import Assistant

router = APIRouter()


def get_assistant(request: Request) -> Assistant:
    return request.app.state.assistant


AssistantDependency = Annotated[Assistant, Depends(get_assistant)]
DemoUser = Annotated[str, Header(alias="X-Demo-User")]


def call_service(operation, *args):
    try:
        return operation(*args)
    except UnknownUserError as error:
        raise HTTPException(401, str(error)) from error
    except IngestForbiddenError as error:
        raise HTTPException(403, str(error)) from error
    except StorageError as error:
        raise HTTPException(503, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(503, "模型服务调用失败，请检查 Ollama 配置。") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest(body: IngestRequest, assistant: AssistantDependency, x_demo_user: DemoUser = "alice"):
    """同步 upsert JSON 文档；相同 id 替换、版本自动管理。演示用户 helen 可写入。"""
    return call_service(assistant.ingest, body, x_demo_user)


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(body: QueryRequest, assistant: AssistantDependency, x_demo_user: DemoUser = "alice"):
    return call_service(assistant.ask, body.question, x_demo_user, body.session_id)


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(assistant: AssistantDependency):
    return assistant.health()


@router.get("/", include_in_schema=False)
def root():
    return {"message": "打开 /docs 测试 /ingest、/query 和 /health"}
