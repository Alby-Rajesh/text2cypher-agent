import json
import re
from functools import lru_cache
from pathlib import Path
from string import Template

import yaml

_FENCE = re.compile(r"```(?:cypher)?\s*(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)


@lru_cache
def load_prompts(path: Path = Path("configs/prompts.yaml")) -> dict[str, Template]:
    return {k: Template(v) for k, v in yaml.safe_load(path.read_text(encoding="utf-8")).items()}


def cypher_messages(
    prompts: dict[str, Template],
    schema: str,
    question: str,
    examples: list[dict] | None = None,
    history: list[dict] | None = None,
    failed: tuple[str, str] | None = None,
) -> list[dict]:
    system = prompts["cypher_system"].substitute(schema=schema.strip())
    if examples:
        shots = "\n\n".join(f"Question: {e['question']}\nCypher: {e['cypher']}" for e in examples)
        system += "\n" + prompts["cypher_examples"].substitute(examples=shots)
    if history:
        turns = "\n\n".join(f"Question: {h['question']}\nCypher: {h['cypher']}" for h in history if h.get("cypher"))
        if turns:
            system += "\n" + prompts["cypher_history"].substitute(history=turns)
    messages = [{"role": "system", "content": system.strip()}, {"role": "user", "content": question.strip()}]
    if failed:
        cypher, error = failed
        messages += [
            {"role": "assistant", "content": cypher},
            {"role": "user", "content": prompts["cypher_correction"].substitute(cypher=cypher, error=error).strip()},
        ]
    return messages


def answer_messages(prompts: dict[str, Template], question: str, rows: list[dict], max_chars: int = 6000) -> list[dict]:
    payload = json.dumps(rows, ensure_ascii=False, default=str)[:max_chars]
    return [
        {"role": "system", "content": prompts["answer_system"].template.strip()},
        {
            "role": "user",
            "content": prompts["answer_user"].substitute(question=question, count=len(rows), rows=payload).strip(),
        },
    ]


def extract_cypher(text: str) -> str:
    text = _THINK.sub("", text or "").strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    text = re.sub(r"^\s*cypher\s*:\s*", "", text.strip(), flags=re.I)
    return text.strip().rstrip(";").strip()
