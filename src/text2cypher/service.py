import logging

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .agent import Dependencies, Text2CypherAgent
from .cache import QueryCache
from .examples import ExampleStore
from .graphdb import GraphDB
from .prompts import load_prompts
from .settings import LLMSettings, Settings

log = logging.getLogger(__name__)


def chat_model(settings: LLMSettings, model: str, max_tokens: int | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        temperature=settings.temperature,
        max_tokens=max_tokens or settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        extra_body=settings.extra or None,
    )


def redis_backends(settings: Settings) -> tuple[object | None, BaseCheckpointSaver]:
    if not settings.redis.url:
        log.warning("Redis disabled: no cache, in-process conversation memory")
        return None, InMemorySaver()
    import redis
    from langgraph.checkpoint.redis import RedisSaver

    client = redis.Redis.from_url(settings.redis.url)
    saver = RedisSaver(
        redis_client=client, ttl={"default_ttl": settings.redis.memory_ttl_minutes, "refresh_on_read": True}
    )
    saver.setup()
    return client, saver


def build_agent(settings: Settings) -> Text2CypherAgent:
    settings.configure_tracing()
    client, checkpointer = redis_backends(settings)
    deps = Dependencies(
        db=GraphDB(settings.neo4j),
        cypher_llm=chat_model(settings.llm, settings.llm.cypher_model),
        answer_llm=chat_model(settings.llm, settings.llm.answer_model),
        examples=ExampleStore.from_jsonl(settings.agent.examples_path),
        cache=QueryCache(client, settings.neo4j.database, settings.redis.cache_ttl),
        prompts=load_prompts(settings.agent.prompts_path),
        settings=settings.agent,
    )
    log.info("agent ready: database=%s examples=%d", settings.neo4j.database, len(deps.examples.examples))
    return Text2CypherAgent(deps, checkpointer)
