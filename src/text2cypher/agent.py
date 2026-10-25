import logging
import operator
import uuid
from dataclasses import dataclass
from string import Template
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel

from .cache import QueryCache
from .examples import ExampleStore
from .graphdb import GraphDB
from .prompts import answer_messages, cypher_messages, extract_cypher
from .settings import AgentSettings

log = logging.getLogger(__name__)


class Turn(TypedDict):
    question: str
    cypher: str
    answer: str


class State(TypedDict, total=False):
    question: str
    skip_answer: bool
    history: Annotated[list[Turn], operator.add]
    examples: list[dict]
    cypher: str
    error: str | None
    attempts: int
    rows: list[dict]
    answer: str
    cached: bool


class AgentResult(BaseModel):
    thread_id: str
    question: str
    answer: str
    cypher: str
    rows: list[dict]
    attempts: int
    cached: bool
    error: str | None = None


@dataclass
class Dependencies:
    db: GraphDB
    cypher_llm: BaseChatModel
    answer_llm: BaseChatModel
    examples: ExampleStore
    cache: QueryCache
    prompts: dict[str, Template]
    settings: AgentSettings


def build_graph(deps: Dependencies, checkpointer: BaseCheckpointSaver | None = None):
    s = deps.settings

    def prepare(state: State) -> State:
        reset: State = {"cypher": "", "error": None, "attempts": 0, "rows": [], "answer": "", "cached": False}
        if not state.get("history") and not state.get("skip_answer"):
            hit = deps.cache.get(state["question"])
            if hit:
                return {**reset, **hit, "cached": True}
        return {**reset, "examples": deps.examples.search(state["question"], s.few_shot_k)}

    def generate(state: State) -> State:
        failed = (state["cypher"], state["error"]) if state.get("error") and state.get("cypher") else None
        messages = cypher_messages(
            deps.prompts,
            deps.db.schema,
            state["question"],
            examples=state.get("examples"),
            history=(state.get("history") or [])[-s.history_turns :] if s.history_turns else None,
            failed=failed,
        )
        reply = deps.cypher_llm.invoke(messages)
        return {"cypher": extract_cypher(str(reply.content)), "attempts": state.get("attempts", 0) + 1}

    def validate(state: State) -> State:
        return {"error": deps.db.validate(state["cypher"])}

    def execute(state: State) -> State:
        try:
            return {"rows": deps.db.run(state["cypher"]), "error": None}
        except Neo4jError as exc:
            return {"error": exc.message or str(exc)}

    def answer(state: State) -> State:
        reply = deps.answer_llm.invoke(answer_messages(deps.prompts, state["question"], state["rows"]))
        result = {"answer": str(reply.content).strip()}
        if not state.get("history"):
            deps.cache.set(state["question"], {"cypher": state["cypher"], "rows": state["rows"], **result})
        return result

    def give_up(state: State) -> State:
        return {
            "answer": "I couldn't turn that question into a valid query for this database. Try rephrasing it with names or properties from the data."
        }

    def remember(state: State) -> State:
        return {
            "history": [
                {"question": state["question"], "cypher": state.get("cypher", ""), "answer": state.get("answer", "")}
            ]
        }

    def after_prepare(state: State) -> Literal["remember", "generate"]:
        return "remember" if state.get("cached") else "generate"

    def after_validate(state: State) -> Literal["execute", "generate", "give_up"]:
        if state.get("error") is None:
            return "execute"
        return "generate" if state["attempts"] < s.max_attempts else "give_up"

    def after_execute(state: State) -> Literal["answer", "remember", "generate", "give_up"]:
        if state.get("error") is None:
            return "remember" if state.get("skip_answer") else "answer"
        return "generate" if state["attempts"] < s.max_attempts else "give_up"

    graph = StateGraph(State)
    for name, fn in [
        ("prepare", prepare),
        ("generate", generate),
        ("validate", validate),
        ("execute", execute),
        ("answer", answer),
        ("give_up", give_up),
        ("remember", remember),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", after_prepare)
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", after_validate)
    graph.add_conditional_edges("execute", after_execute)
    graph.add_edge("answer", "remember")
    graph.add_edge("give_up", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class Text2CypherAgent:
    def __init__(self, deps: Dependencies, checkpointer: BaseCheckpointSaver | None = None):
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)

    def ask(self, question: str, thread_id: str | None = None, skip_answer: bool = False) -> AgentResult:
        thread_id = thread_id or uuid.uuid4().hex
        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": "text2cypher",
            "metadata": {"thread_id": thread_id},
        }
        state = self.graph.invoke({"question": question, "skip_answer": skip_answer}, config)
        return AgentResult(
            thread_id=thread_id,
            question=question,
            answer=state.get("answer", ""),
            cypher=state.get("cypher", ""),
            rows=state.get("rows", []),
            attempts=state.get("attempts", 0),
            cached=state.get("cached", False),
            error=state.get("error"),
        )
