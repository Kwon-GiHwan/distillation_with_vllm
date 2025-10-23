"""Tests for `app.eval.eval_bench`."""
from typer.testing import CliRunner
import yaml
import orjson
from app.eval import eval_bench

from app.cli import app

runner = CliRunner()

def test_run_eval_bench(mocker, tmp_path):
    """Verify `run_eval_bench` returns a valid accuracy score."""
    # 1. Mock the evaluate_task function
    mock_evaluate = mocker.patch("app.eval.eval_bench.evaluate_task")
    mock_evaluate.return_value = 0.5

    # 2. Mock model and tokenizer loading
    mocker.patch("app.eval.eval_bench.AutoModelForCausalLM.from_pretrained")
    mocker.patch("app.eval.eval_bench.AutoTokenizer.from_pretrained")

    # 3. Create a dummy config
    config = {
        "model_path": "dummy-model",
        "tasks": "piqa,hellaswag",
        "batch_size": 1
    }

    # 4. Run the eval-bench function directly
    results = eval_bench.run_eval_bench(**config)

    # 5. Check that evaluate_task was called for each task
    assert mock_evaluate.call_count == 2

    # 6. Check the returned results
    assert results["piqa"] == 0.5
    assert results["hellaswag"] == 0.5
