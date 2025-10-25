import json
import typer
from src.config import ConfigLoader
from src.pipeline import Pipeline


app = typer.Typer(add_completion=False)


@app.command()
def sft(config: str = "configs/config.yaml"):
    cfg = ConfigLoader.load(config)
    Pipeline(cfg).sft()


@app.command()
def kd_collect(config: str = "configs/config.yaml"):
    cfg = ConfigLoader.load(config)
    path = Pipeline(cfg).kd_collect()
    typer.echo(f"Saved KD dataset → {path}")


@app.command()
def kd_train(config: str = "configs/config.yaml"):
    cfg = ConfigLoader.load(config)
    Pipeline(cfg).kd_train()


@app.command()
def eval_all(config: str = "configs/config.yaml"):
    cfg = ConfigLoader.load(config)
    res = Pipeline(cfg).eval_all()
    typer.echo(json.dumps(res, indent=2))


@app.command()
def run_all(config: str = "configs/config.yaml"):
    cfg = ConfigLoader.load(config)
    res = Pipeline(cfg).run_all()
    typer.echo(json.dumps(res, indent=2))


if __name__ == "__main__":
    app()