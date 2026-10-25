from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import CypherSyntaxError

from text2cypher.agent import AgentResult
from text2cypher.api import create_app
from text2cypher.cache import QueryCache, normalize
from text2cypher.evaluation import Case, same_result, summarize
from text2cypher.examples import ExampleStore
from text2cypher.graphdb import GraphDB, format_schema
from text2cypher.prompts import cypher_messages, extract_cypher
from text2cypher.settings import Neo4jSettings


@pytest.mark.parametrize(
    "raw",
    [
        "MATCH (n) RETURN n",
        "```cypher\nMATCH (n) RETURN n;\n```",
        "Cypher: MATCH (n) RETURN n",
        "<think>plan</think>```\nMATCH (n) RETURN n\n```",
    ],
)
def test_extract_cypher(raw):
    assert extract_cypher(raw) == "MATCH (n) RETURN n"


def test_training_and_inference_share_the_zero_shot_prompt(prompts):
    messages = cypher_messages(prompts, "SCHEMA", "How many movies?")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "SCHEMA" in messages[0]["content"] and messages[1]["content"] == "How many movies?"


def test_few_shot_and_history_are_added_to_system_prompt(prompts):
    messages = cypher_messages(
        prompts,
        "SCHEMA",
        "q",
        examples=[{"question": "How many people?", "cypher": "MATCH (p:Person) RETURN count(p)"}],
        history=[{"question": "earlier", "cypher": "MATCH (m:Movie) RETURN m", "answer": "a"}],
    )
    assert "MATCH (p:Person) RETURN count(p)" in messages[0]["content"]
    assert "MATCH (m:Movie) RETURN m" in messages[0]["content"]


def test_format_schema():
    schema = format_schema(
        [{"nodeLabels": ["Movie"], "propertyName": "title", "propertyTypes": ["String"]}],
        [{"relType": ":`ACTED_IN`", "propertyName": "roles", "propertyTypes": ["StringArray"]}],
        [{"source": "Person", "type": "ACTED_IN", "target": "Movie"}],
    )
    assert "- **Movie**\n  - `title`: STRING" in schema
    assert "- **ACTED_IN**\n  - `roles`: STRING LIST" in schema
    assert schema.endswith("(:Person)-[:ACTED_IN]->(:Movie)")


class ExplainDriver:
    def __init__(self, query_type="r", error=None):
        self.query_type, self.error = query_type, error

    def execute_query(self, query, **kwargs):
        if self.error:
            raise self.error
        return [], SimpleNamespace(query_type=self.query_type), []


def test_validate_rejects_writes_and_syntax_errors():
    settings = Neo4jSettings()
    assert GraphDB(settings, ExplainDriver("r")).validate("MATCH (n) RETURN n") is None
    assert "read-only" in GraphDB(settings, ExplainDriver("rw")).validate("MATCH (n) SET n.x = 1")
    error = (
        CypherSyntaxError._hydrate_gql(message="Invalid input 'MATC'")
        if hasattr(CypherSyntaxError, "_hydrate_gql")
        else CypherSyntaxError("Invalid input 'MATC'")
    )
    assert GraphDB(settings, ExplainDriver(error=error)).validate("MATC (n)") is not None
    assert GraphDB(settings, ExplainDriver()).validate("  ") == "The query is empty."


def test_example_store_ranks_by_keyword_overlap():
    store = ExampleStore(
        [
            {"question": "How many movies were released in 1999?", "cypher": "A"},
            {"question": "Who directed Inception?", "cypher": "B"},
        ]
    )
    assert [e["cypher"] for e in store.search("which movies came out in 1999", 1)] == ["A"]
    assert ExampleStore([]).search("anything", 3) == []


def test_cache_normalizes_questions_and_survives_backend_errors():
    class Broken:
        def get(self, key):
            raise ConnectionError("down")

        def set(self, key, value, ex=None):
            raise ConnectionError("down")

    assert normalize("  Who ACTED in   The Matrix?? ") == "who acted in the matrix"
    cache = QueryCache(Broken(), "movies", 60)
    assert cache.get("q") is None
    cache.set("q", {"a": 1})


def test_result_comparison_ignores_column_names_and_unordered_row_order():
    gold = [{"name": "A", "n": 1}, {"name": "B", "n": 2}]
    pred = [{"count": 2, "person": "B"}, {"count": 1, "person": "A"}]
    assert same_result(gold, pred, ordered=False)
    assert not same_result(gold, pred, ordered=True)
    assert not same_result(gold, pred[:1], ordered=False)


def test_summary_metrics():
    cases = [
        Case("q", "g", valid=True, correct=True, attempts=1),
        Case("q", "g", valid=True, correct=True, attempts=2),
        Case("q", "g", attempts=3),
    ]
    summary = summarize(cases)
    assert summary["execution_accuracy"] == round(2 / 3, 4)
    assert summary["first_try_accuracy"] == round(1 / 3, 4)
    assert summary["mean_attempts"] == 2.0


def test_api_query_and_validation():
    class Agent:
        def ask(self, question, thread_id=None):
            return AgentResult(
                thread_id=thread_id or "t1",
                question=question,
                answer="ok",
                cypher="MATCH (n) RETURN n",
                rows=[],
                attempts=1,
                cached=False,
            )

    with TestClient(create_app(lambda: Agent())) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        body = client.post("/v1/query", json={"question": "How many movies?", "thread_id": "abc"}).json()
        assert body["answer"] == "ok" and body["thread_id"] == "abc"
        assert client.post("/v1/query", json={"question": ""}).status_code == 422
