import typer
from typing_extensions import Annotated
from app.utils import load_yaml
from app.collect import run_collect
from app.train_sft import run_train_sft
from app.train_kd import run_train_kd
from eval_bench import run_eval_bench
from eval_similarity import run_eval_similarity

app = typer.Typer(help="distillation_with_vllm CLI")

DEFAULT_CONFIGS = {
    "collect": "configs/collect.yaml",
    "train_sft": "configs/train_sft.yaml",
    "train_kd": "configs/train_kd.yaml",
}

@app.command()
def collect(
    config_path: Annotated[
        str, typer.Option(help="Path to the collect config file")
    ] = DEFAULT_CONFIGS["collect"]
):
    """
    Run teacher inference and build distillation pairs
    """
    print(f"Running collect with config: {config_path}")
    cfg = load_yaml(config_path)
    run_collect(cfg)

@app.command()
def train_sft(
    config_path: Annotated[
        str, typer.Option(help="Path to the train_sft config file")
    ] = DEFAULT_CONFIGS["train_sft"]
):
    """
    Run supervised fine-tuning (SFT)
    """
    print(f"Running train_sft with config: {config_path}")
    cfg = load_yaml(config_path)
    run_train_sft(cfg)

@app.command()
def train_kd(
    config_path: Annotated[
        str, typer.Option(help="Path to the train_kd config file")
    ] = DEFAULT_CONFIGS["train_kd"]
):
    """
    Run knowledge distillation (CE + KL)
    """
    print(f"Running train_kd with config: {config_path}")
    cfg = load_yaml(config_path)
    run_train_kd(cfg)

@app.command()
def pipeline(
    collect_config: Annotated[
        str, typer.Option(help="Path to the collect config file")
    ] = DEFAULT_CONFIGS["collect"],
    train_sft_config: Annotated[
        str, typer.Option(help="Path to the train_sft config file")
    ] = DEFAULT_CONFIGS["train_sft"],
    train_kd_config: Annotated[
        str, typer.Option(help="Path to the train_kd config file")
    ] = DEFAULT_CONFIGS["train_kd"],
    skip_collect: bool = typer.Option(False, help="Skip the collect step"),
    skip_train_sft: bool = typer.Option(False, help="Skip the train_sft step"),
    skip_train_kd: bool = typer.Option(False, help="Skip the train_kd step"),
    skip_eval: bool = typer.Option(False, help="Skip the evaluation step"),
):
    """
    Run the full pipeline: collect -> train_sft -> train_kd -> eval
    """
    print("--- Running Pipeline ---")

    # Load configs
    collect_cfg = load_yaml(collect_config)
    train_sft_cfg = load_yaml(train_sft_config)
    train_kd_cfg = load_yaml(train_kd_config)

    # 1. Collect
    if not skip_collect:
        print(f"Running collect with config: {collect_config}")
        run_collect(collect_cfg)

    # 2. Train SFT
    if not skip_train_sft:
        print(f"Running train_sft with config: {train_sft_config}")
        run_train_sft(train_sft_cfg)

    # 3. Train KD
    if not skip_train_kd:
        print(f"Running train_kd with config: {train_kd_config}")
        run_train_kd(train_kd_cfg)

    # 4. Evaluation
    if not skip_eval:
        print("--- Running Evaluation ---")
        teacher_model_path = train_kd_cfg["teacher_model_name_or_path"]
        sft_model_path = train_sft_cfg["output_dir"]
        kd_model_path = train_kd_cfg["output_dir"]

        print("\n--- Evaluating SFT model ---")
        run_eval_bench(model_path=sft_model_path)
        run_eval_similarity(teacher_path=teacher_model_path, student_path=sft_model_path)

        print("\n--- Evaluating KD model ---")
        run_eval_bench(model_path=kd_model_path)
        run_eval_similarity(teacher_path=teacher_model_path, student_path=kd_model_path)

    print("--- Pipeline Finished ---")
