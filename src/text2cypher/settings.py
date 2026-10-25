import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: SecretStr = SecretStr("token-is-ignored")
    cypher_model: str = "text2cypher"
    answer_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(512, gt=0)
    timeout: float = Field(60, gt=0)
    max_retries: int = Field(2, ge=0)


class Neo4jSettings(BaseModel):
    uri: str = "neo4j+s://demo.neo4jlabs.com:7687"
    user: str = "movies"
    password: SecretStr = SecretStr("movies")
    database: str = "movies"
    query_timeout: float = Field(15, gt=0)
    row_limit: int = Field(50, ge=1)


class RedisSettings(BaseModel):
    url: str | None = "redis://localhost:6379/0"
    cache_ttl: int = Field(86400, ge=0)
    memory_ttl_minutes: int = Field(1440, ge=1)


class AgentSettings(BaseModel):
    max_attempts: int = Field(3, ge=1)
    few_shot_k: int = Field(4, ge=0)
    history_turns: int = Field(3, ge=0)
    examples_path: Path = Path("data/examples.jsonl")
    prompts_path: Path = Path("configs/prompts.yaml")


class LangSmithSettings(BaseModel):
    api_key: SecretStr | None = None
    project: str = "text2cypher-agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="T2C_", env_nested_delimiter="__", env_file=".env", extra="ignore")

    llm: LLMSettings = LLMSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    redis: RedisSettings = RedisSettings()
    agent: AgentSettings = AgentSettings()
    langsmith: LangSmithSettings = LangSmithSettings()

    def configure_tracing(self) -> None:
        if self.langsmith.api_key:
            os.environ.setdefault("LANGSMITH_TRACING", "true")
            os.environ.setdefault("LANGSMITH_API_KEY", self.langsmith.api_key.get_secret_value())
            os.environ.setdefault("LANGSMITH_PROJECT", self.langsmith.project)


@lru_cache
def get_settings() -> Settings:
    return Settings()
