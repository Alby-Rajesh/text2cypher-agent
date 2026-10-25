import hashlib
import json
import logging
import re
from typing import Protocol

log = logging.getLogger(__name__)


class KeyValue(Protocol):
    def get(self, key: str) -> bytes | str | None: ...
    def set(self, key: str, value: str, ex: int | None = None) -> object: ...


def normalize(question: str) -> str:
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", question.lower())).strip()


class QueryCache:
    def __init__(self, client: KeyValue | None, namespace: str, ttl: int):
        self.client, self.namespace, self.ttl = client, namespace, ttl

    def key(self, question: str) -> str:
        return f"t2c:{self.namespace}:{hashlib.sha256(normalize(question).encode()).hexdigest()}"

    def get(self, question: str) -> dict | None:
        if self.client is None:
            return None
        try:
            raw = self.client.get(self.key(question))
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.warning("cache read failed: %s", exc)
            return None

    def set(self, question: str, value: dict) -> None:
        if self.client is None:
            return
        try:
            self.client.set(self.key(question), json.dumps(value, default=str), ex=self.ttl or None)
        except Exception as exc:
            log.warning("cache write failed: %s", exc)
