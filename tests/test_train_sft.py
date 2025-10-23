"""Tests for `app.train_sft`."""
import json
import yaml
from pathlib import Path
from app import train_sft

def test_run_train_sft(tmp_path):
    """Verify `run_train_sft` creates a model artifact."""
    # 1. Create dummy dataset
    dataset_path = tmp_path / "dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps({"prompt": "Hello", "text": "World"}))

    # 2. Create dummy config
    output_dir = tmp_path / "artifacts/sft_model"
    final_dir = output_dir / "final"
    config = {
        "student_model": "sshleifer/tiny-gpt2",
        "data_path": str(dataset_path),
        "out_dir": str(output_dir),
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "lr": 1e-4,
        "scheduler": "cosine",
        "warmup": 0.1,
        "logging_steps": 1,
        "max_len": 128,
        "final_dir": str(final_dir),
        "bf16": False
    }

    # 3. Run train_sft function directly
    train_sft.run_train_sft(config)

    # 4. Validate that the final model was saved
    assert final_dir.is_dir()
    assert (final_dir / "pytorch_model.bin").exists()
    assert (final_dir / "config.json").exists()
    assert (final_dir / "tokenizer.json").exists()

    # 5. Check for trainer state
    assert (output_dir / "trainer_state.json").exists()
