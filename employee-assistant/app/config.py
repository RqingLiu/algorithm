"""Config：集中读取运行参数；不会在导入时访问模型或知识库。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    policies_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("POLICIES_PATH", str(PROJECT_ROOT / "policies.json"))
        )
    )
    model_mode: str = field(default_factory=lambda: os.getenv("MODEL_MODE", "demo"))
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", ""))
    cache_ttl: int = 300
    session_ttl: int = 3600
