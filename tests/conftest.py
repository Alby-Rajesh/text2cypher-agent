from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from text2cypher.agent import Dependencies, Text2CypherAgent
from text2cypher.cache import QueryCache
from text2cypher.examples import ExampleStore
from text2cypher.prompts import load_prompts
from text2cypher.settings import AgentSettings

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "Node properties:\n- **Movie**\n  - `title`: STRING\n- **Person**\n  - `name`: STRING\nThe relationships:\n(:Person)-[:ACTED_IN]->(:Movie)"


class FakeDB:
    def __init__(self, errors: dict[str, str] | None = None, rows: list[dict] | None = None):
        self.schema = SCHEMA
        self.errors = errors or {}
        self.rows = rows if rows is not None else [{"name": "Keanu Reeves"}]
        self.validated: list[str] = []
        self.executed: list[str] = []

    def validate(self, cypher: str) -> str | None:
        self.validated.append(cypher)
        return self.errors.get(cypher)

    def run(self, cypher: str) -> list[dict]:
        self.executed.append(cypher)
        return self.rows


class DictRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture
def prompts():
    return load_prompts(ROOT / "configs/prompts.yaml")


@pytest.fixture
def make_agent(prompts):
    def factory(
        cypher_replies,
        answer_replies=("Keanu Reeves acted in The Matrix.",),
        db=None,
        cache_client=None,
        examples=(),
        **agent_kwargs,
    ):
        deps = Dependencies(
            db=db or FakeDB(),
            cypher_llm=FakeListChatModel(responses=list(cypher_replies)),
            answer_llm=FakeListChatModel(responses=list(answer_replies)),
            examples=ExampleStore(list(examples)),
            cache=QueryCache(cache_client, "movies", ttl=60),
            prompts=prompts,
            settings=AgentSettings(**agent_kwargs),
        )
        return Text2CypherAgent(deps)

    return factory
