import argparse
import json
import logging
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset

from text2cypher.prompts import cypher_messages, load_prompts

log = logging.getLogger("prepare")


def to_example(row: dict, prompts) -> dict:
    return {
        "prompt": cypher_messages(prompts, row["schema"], row["question"]),
        "completion": [{"role": "assistant", "content": row["cypher"].strip()}],
    }


def clean(split: Dataset, max_schema_chars: int) -> Dataset:
    split = split.filter(lambda r: r["schema"] and r["cypher"] and len(r["schema"]) <= max_schema_chars)
    seen: set[tuple[str, str]] = set()

    def first(row: dict) -> bool:
        key = (row["question"].strip().lower(), row["cypher"].strip())
        if key in seen:
            return False
        seen.add(key)
        return True

    return split.filter(first)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build prompt-completion SFT data in the agent's exact prompt format.")
    ap.add_argument("--config", default="finetune/config.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())["data"]
    prompts = load_prompts(Path(args.prompts))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(cfg["dataset"])
    train = clean(raw["train"], cfg["max_schema_chars"]).shuffle(seed=cfg["seed"])
    test = clean(raw["test"], cfg["max_schema_chars"]).shuffle(seed=cfg["seed"])
    splits = {
        "train": train.select(range(min(cfg["train_samples"], len(train)))),
        "eval": test.select(range(min(cfg["eval_samples"], len(test)))),
    }
    for name, split in splits.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(to_example(row, prompts), ensure_ascii=False) + "\n")
        log.info("%s: %d examples -> %s", name, len(split), path)


if __name__ == "__main__":
    main()
