"""Tests for `app.collect`."""
import orjson
import yaml
from app import collect

def test_run_collect(mocker, tmp_path):
    """Verify `run_collect` generates data correctly."""
    # 1. Create dummy config
    config = {
        "dataset": {"name": "alpaca", "n_samples": 1},
        "hosting": {
            "model": "sshleifer/tiny-gpt2",
            "dtype": "auto",
            "quantization": None,
            "gpu_memory_utilization": 0.9
        },
        "sampling": {
            "temperature": 0.9,
            "top_p": 1.0,
            "max_tokens": 10,
            "logprobs": 1,
        },
        "output": {"path": str(tmp_path / "distilled.jsonl")}
    }

    # 2. Mock the vLLM engine
    mock_engine = mocker.patch("app.collect.VLLMEngine")
    mock_engine.return_value.generate.return_value = [
        {"prompt": "Hello", "text": "World", "logprobs": [-0.1, -0.2]}
    ]

    # 3. Mock the dataset loader
    mock_dataset = mocker.MagicMock()
    mock_dataset.select.return_value = [{"instruction": "Hello", "input": ""}]
    mocker.patch("app.collect.load_dataset", return_value=mock_dataset)

    # 4. Run collect function directly
    collect.run_collect(config)

    # 5. Validate output file
    output_path = config["output"]["path"]
    with open(output_path, "rb") as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = orjson.loads(lines[0])
        assert record["prompt"] == "Hello"
        assert record["text"] == "World"
        assert record["logprobs"] == [-0.1, -0.2]
