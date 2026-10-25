import json
import logging
import uuid
from pathlib import Path

import typer

from .settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Agentic Text-to-Cypher over Neo4j.")


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "neo4j", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command(help="Chat with the database in the terminal; follow-up questions keep context.")
def chat(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    from .service import build_agent

    _logging(verbose)
    agent, thread = build_agent(get_settings()), uuid.uuid4().hex
    typer.echo(f"Connected to '{get_settings().neo4j.database}'. Ask a question, or press Ctrl+D to quit.")
    while True:
        try:
            question = typer.prompt(">", prompt_suffix=" ")
        except (EOFError, typer.Abort):
            break
        result = agent.ask(question, thread)
        typer.secho(result.cypher, fg="cyan")
        typer.echo(f"{result.answer}\n")


@app.command(help="Answer one question and print the result as JSON.")
def ask(question: str, verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    from .service import build_agent

    _logging(verbose)
    typer.echo(json.dumps(build_agent(get_settings()).ask(question).model_dump(), indent=2, default=str))


@app.command(help="Serve the HTTP API.")
def serve(host: str = "0.0.0.0", port: int = 8080, workers: int = 1) -> None:
    import uvicorn

    _logging(False)
    uvicorn.run("text2cypher.api:create_app", factory=True, host=host, port=port, workers=workers)


@app.command(help="Build the few-shot example store from the dataset's train split.")
def examples(limit: int = typer.Option(2000, help="Maximum examples to keep.")) -> None:
    from .examples import load_split, write_jsonl

    settings = get_settings()
    rows = load_split(settings.neo4j.dataset_name, "train", limit)
    write_jsonl(rows, settings.agent.examples_path)
    typer.echo(f"wrote {len(rows)} examples to {settings.agent.examples_path}")


@app.command(help="Measure execution accuracy on the dataset's test split.")
def evaluate(
    limit: int = typer.Option(200, help="Test questions to evaluate."),
    out: Path = typer.Option(Path("output/eval"), help="Where to write results."),
    workers: int = typer.Option(4, help="Questions evaluated in parallel."),
    langsmith: bool = typer.Option(False, "--langsmith", help="Also log the run as a LangSmith experiment."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    from .evaluation import run_eval, run_langsmith_eval
    from .examples import load_split
    from .service import build_agent

    _logging(verbose)
    settings = get_settings()
    agent = build_agent(settings)
    test = load_split(settings.neo4j.dataset_name, "test", limit)
    summary = run_eval(agent, agent.deps.db, test, out / settings.llm.cypher_model.replace("/", "_"), workers)
    typer.echo(json.dumps(summary, indent=2))
    if langsmith:
        run_langsmith_eval(
            agent, agent.deps.db, test, f"text2cypher-{settings.neo4j.dataset_name}", settings.llm.cypher_model
        )


if __name__ == "__main__":
    app()
