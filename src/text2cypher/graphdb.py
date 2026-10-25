import logging
from collections import defaultdict
from functools import cached_property
from typing import Any

from langsmith import traceable
from neo4j import Driver, GraphDatabase, RoutingControl, unit_of_work
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node, Path, Relationship

from .settings import Neo4jSettings

log = logging.getLogger(__name__)

NODE_PROPS = "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes RETURN nodeLabels, propertyName, propertyTypes"
REL_PROPS = "CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes RETURN relType, propertyName, propertyTypes"
PATTERNS = "CALL db.schema.visualization() YIELD relationships RETURN relationships"


def _type(types: list[str] | None) -> str:
    return (types or ["ANY"])[0].replace("Array", " LIST").upper()


def format_schema(node_rows: list[dict], rel_rows: list[dict], patterns: list[dict]) -> str:
    nodes: dict[str, list[str]] = defaultdict(list)
    for row in node_rows:
        label = ":".join(row["nodeLabels"] or [])
        if label and row["propertyName"]:
            nodes[label].append(f"  - `{row['propertyName']}`: {_type(row['propertyTypes'])}")
    rels: dict[str, list[str]] = defaultdict(list)
    for row in rel_rows:
        rel = (row["relType"] or "").strip(":`")
        if rel and row["propertyName"]:
            rels[rel].append(f"  - `{row['propertyName']}`: {_type(row['propertyTypes'])}")
    lines = ["Node properties:"]
    for label in sorted(nodes):
        lines += [f"- **{label}**", *sorted(set(nodes[label]))]
    if rels:
        lines.append("Relationship properties:")
        for rel in sorted(rels):
            lines += [f"- **{rel}**", *sorted(set(rels[rel]))]
    lines.append("The relationships:")
    lines += sorted(
        {f"(:{p['source']})-[:{p['type']}]->(:{p['target']})" for p in patterns if p["source"] and p["target"]}
    )
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Node):
        return dict(value.items())
    if isinstance(value, Relationship):
        return {"type": value.type, **dict(value.items())}
    if isinstance(value, Path):
        return [to_jsonable(n) for n in value.nodes]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "iso_format"):
        return value.iso_format()
    return value


class GraphDB:
    def __init__(self, settings: Neo4jSettings, driver: Driver | None = None):
        self.settings = settings
        self.driver = driver or GraphDatabase.driver(
            settings.uri, auth=(settings.user, settings.password.get_secret_value())
        )

    def _query(self, cypher: str) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            cypher, database_=self.settings.database, routing_=RoutingControl.READ
        )
        return [r.data() for r in records]

    def _patterns(self) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            PATTERNS, database_=self.settings.database, routing_=RoutingControl.READ
        )
        return [
            {
                "source": next(iter(r.start_node.labels), None),
                "type": r.type,
                "target": next(iter(r.end_node.labels), None),
            }
            for record in records
            for r in record["relationships"]
        ]

    @cached_property
    def schema(self) -> str:
        return format_schema(self._query(NODE_PROPS), self._query(REL_PROPS), self._patterns())

    @traceable(name="validate_cypher", run_type="tool")
    def validate(self, cypher: str) -> str | None:
        if not cypher.strip():
            return "The query is empty."
        try:
            _, summary, _ = self.driver.execute_query(
                f"EXPLAIN {cypher}", database_=self.settings.database, routing_=RoutingControl.READ
            )
        except Neo4jError as exc:
            return exc.message or str(exc)
        if summary.query_type != "r":
            return "Only read-only queries are allowed; the query would modify the database."
        return None

    @traceable(name="run_cypher", run_type="tool")
    def run(self, cypher: str) -> list[dict]:
        @unit_of_work(timeout=self.settings.query_timeout)
        def work(tx):
            result = tx.run(cypher)
            return [to_jsonable(r.data()) for r in result.fetch(self.settings.row_limit)]

        with self.driver.session(database=self.settings.database, default_access_mode="READ") as session:
            return session.execute_read(work)

    def close(self) -> None:
        self.driver.close()
