from conftest import DictRedis, FakeDB

GOOD = "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name AS name"
BAD = "MATCH (p:Actor)-[:ACTED_IN]->(m:Movie) RETURN p.name"


def test_answers_on_first_valid_query(make_agent):
    agent = make_agent([f"```cypher\n{GOOD};\n```"])
    result = agent.ask("Who acted in The Matrix?")
    assert result.cypher == GOOD
    assert result.attempts == 1
    assert result.rows == [{"name": "Keanu Reeves"}]
    assert result.answer == "Keanu Reeves acted in The Matrix."
    assert result.error is None


def test_self_corrects_after_validation_error(make_agent):
    db = FakeDB(errors={BAD: "Unknown label Actor"})
    agent = make_agent([BAD, GOOD], db=db)
    result = agent.ask("Who acted in The Matrix?")
    assert result.cypher == GOOD
    assert result.attempts == 2
    assert db.validated == [BAD, GOOD]
    assert db.executed == [GOOD]


def test_correction_prompt_includes_failed_query_and_error(make_agent):
    db = FakeDB(errors={BAD: "Unknown label Actor"})
    agent = make_agent([BAD, GOOD], db=db)
    llm = agent.deps.cypher_llm
    seen = []
    original = llm.invoke
    llm.__dict__["invoke"] = lambda messages, *a, **k: seen.append(messages) or original(messages, *a, **k)
    agent.ask("Who acted in The Matrix?")
    assert seen[1][-2] == {"role": "assistant", "content": BAD}
    assert "Unknown label Actor" in seen[1][-1]["content"]


def test_gives_up_after_max_attempts(make_agent):
    db = FakeDB(errors={BAD: "Unknown label Actor"})
    agent = make_agent([BAD, BAD], db=db, max_attempts=2)
    result = agent.ask("Who acted in The Matrix?")
    assert result.attempts == 2
    assert result.error == "Unknown label Actor"
    assert "couldn't" in result.answer
    assert db.executed == []


def test_cache_hit_skips_llm_and_database(make_agent):
    cache = DictRedis()
    first = make_agent([GOOD], cache_client=cache)
    first.ask("Who acted in The Matrix?")
    db = FakeDB()
    second = make_agent([], db=db, cache_client=cache)
    result = second.ask("who acted in the matrix")
    assert result.cached is True
    assert result.cypher == GOOD
    assert db.validated == [] and db.executed == []


def test_follow_up_uses_conversation_history_and_bypasses_cache(make_agent):
    cache = DictRedis()
    agent = make_agent([GOOD, GOOD], answer_replies=["a1", "a2"], cache_client=cache)
    first = agent.ask("Who acted in The Matrix?")
    second = agent.ask("And in which other movies?", first.thread_id)
    state = agent.graph.get_state({"configurable": {"thread_id": first.thread_id}}).values
    assert [t["question"] for t in state["history"]] == ["Who acted in The Matrix?", "And in which other movies?"]
    assert second.cached is False
    assert len(cache.store) == 1


def test_skip_answer_returns_rows_without_answer_llm(make_agent):
    agent = make_agent([GOOD], answer_replies=[])
    result = agent.ask("Who acted in The Matrix?", skip_answer=True)
    assert result.rows == [{"name": "Keanu Reeves"}]
    assert result.answer == ""
