import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import AgentResult, Text2CypherAgent
from .settings import get_settings

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=128)


def create_app(agent_factory: Callable[[], Text2CypherAgent] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent_factory is not None:
            app.state.agent = agent_factory()
        else:
            from .service import build_agent

            app.state.agent = build_agent(get_settings())
        yield
        db = getattr(getattr(app.state.agent, "deps", None), "db", None)
        if db is not None and hasattr(db, "close"):
            db.close()

    app = FastAPI(title="text2cypher-agent", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/query", response_model=AgentResult)
    def query(body: QueryRequest, request: Request) -> AgentResult:
        try:
            return request.app.state.agent.ask(body.question, body.thread_id)
        except Exception as exc:
            log.exception("query failed")
            raise HTTPException(status_code=502, detail=f"upstream failure: {type(exc).__name__}") from exc

    return app
