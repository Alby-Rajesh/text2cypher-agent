import json
import logging
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from neo4j.exceptions import Neo4jError

from .agent import Text2CypherAgent
from .graphdb import GraphDB

log = logging.getLogger(__name__)

_ORDER = re.compile(r"\bORDER\s+BY\b", re.I)


def _canonical(rows: list[dict]) -> list[tuple]:
    return [tuple(sorted(json.dumps(v, sort_keys=True, default=str) for v in row.values())) for row in rows]


def same_result(gold: list[dict], pred: list[dict], ordered: bool) -> bool:
    g, p = _canonical(gold), _canonical(pred)
    return g == p if ordered else Counter(g) == Counter(p)


def normalize_cypher(cypher: str) -> str:
    return re.sub(r"\s+", " ", cypher.strip().rstrip(";")).lower()


@dataclass
class Case:
    question: str
    gold: str
    pred: str = ""
    valid: bool = False
    correct: bool = False
    exact: bool = False
    attempts: int = 0
    latency: float = 0.0
    error: str | None = None


def evaluate_case(agent: Text2CypherAgent, db: GraphDB, example: dict) -> Case | None:
    case = Case(question=example["question"], gold=example["cypher"])
    try:
        gold_rows = db.run(case.gold)
    except Neo4jError as exc:
        log.debug("skipping example with failing gold query: %s", exc)
        return None
    start = time.perf_counter()
    try:
        result = agent.ask(case.question, skip_answer=True)
    except Exception as exc:
        case.error, case.latency = f"{type(exc).__name__}: {exc}", time.perf_counter() - start
        return case
    case.latency = time.perf_counter() - start
    case.pred, case.attempts, case.error = result.cypher, result.attempts, result.error
    case.valid = result.error is None
    case.correct = case.valid and same_result(gold_rows, result.rows, ordered=bool(_ORDER.search(case.gold)))
    case.exact = normalize_cypher(case.pred) == normalize_cypher(case.gold)
    return case


def summarize(cases: list[Case]) -> dict:
    n = len(cases) or 1
    return {
        "examples": len(cases),
        "execution_accuracy": round(sum(c.correct for c in cases) / n, 4),
        "first_try_accuracy": round(sum(c.correct and c.attempts == 1 for c in cases) / n, 4),
        "valid_query_rate": round(sum(c.valid for c in cases) / n, 4),
        "exact_match": round(sum(c.exact for c in cases) / n, 4),
        "mean_attempts": round(statistics.fmean(c.attempts for c in cases), 3) if cases else 0,
        "p50_latency_s": round(statistics.median(c.latency for c in cases), 3) if cases else 0,
    }


def run_eval(agent: Text2CypherAgent, db: GraphDB, examples: list[dict], out_dir: Path, workers: int = 4) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = [c for c in pool.map(lambda e: evaluate_case(agent, db, e), examples) if c is not None]
    summary = summarize(cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases.jsonl").write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_langsmith_eval(
    agent: Text2CypherAgent, db: GraphDB, examples: list[dict], dataset_name: str, experiment: str
) -> None:
    from langsmith import Client, evaluate

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name, description="Neo4j text2cypher-2024v1 test questions")
        client.create_examples(
            dataset_id=dataset.id,
            examples=[{"inputs": {"question": e["question"]}, "outputs": {"cypher": e["cypher"]}} for e in examples],
        )

    def target(inputs: dict) -> dict:
        return agent.ask(inputs["question"], skip_answer=True).model_dump()

    def execution_match(outputs: dict, reference_outputs: dict) -> dict:
        try:
            gold_rows = db.run(reference_outputs["cypher"])
        except Neo4jError:
            return {"key": "execution_match", "score": None}
        ordered = bool(_ORDER.search(reference_outputs["cypher"]))
        ok = outputs.get("error") is None and same_result(gold_rows, outputs.get("rows", []), ordered)
        return {"key": "execution_match", "score": int(ok)}

    def valid_query(outputs: dict) -> dict:
        return {"key": "valid_query", "score": int(outputs.get("error") is None)}

    evaluate(
        target,
        data=dataset_name,
        evaluators=[execution_match, valid_query],
        experiment_prefix=experiment,
        max_concurrency=4,
    )
