"""业务异常：不依赖 FastAPI；Router 将它们映射成 HTTP 状态码。"""


class UnknownUserError(Exception):
    pass


class IngestForbiddenError(Exception):
    pass


class StorageError(Exception):
    pass
