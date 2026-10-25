import json
import re
from pathlib import Path

from rank_bm25 import BM25Plus

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class ExampleStore:
    def __init__(self, examples: list[dict]):
        self.examples = [e for e in examples if e.get("question") and e.get("cypher")]
        self.bm25 = BM25Plus([tokenize(e["question"]) for e in self.examples]) if self.examples else None

    @classmethod
    def from_jsonl(cls, path: Path) -> "ExampleStore":
        if not path.exists():
            return cls([])
        return cls([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    def search(self, question: str, k: int) -> list[dict]:
        if not self.bm25 or k <= 0:
            return []
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.examples[i] for i in ranked[:k]]


def dataset_alias(database: str) -> str:
    return f"neo4jlabs_demo_db_{database}"


def load_split(database: str, split: str, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("neo4j/text2cypher-2024v1", split=split).filter(
        lambda r: r["database_reference_alias"] == dataset_alias(database)
    )
    rows = rows.select(range(min(limit, len(rows)))) if limit else rows
    return [{"question": r["question"], "cypher": r["cypher"]} for r in rows]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
