"""Tests for `app.train_kd`."""
import json
import yaml
import torch
from pathlib import Path
from app import train_kd

def test_run_train_kd(tmp_path):
    """Verify `run_train_kd` executes a training step and saves a checkpoint."""
    # 1. Create dummy dataset with logprobs
    dataset_path = tmp_path / "dataset.jsonl"

    with open(dataset_path, "w") as f:
        record = {
            "prompt": "Q: What is the capital of France?",
            "text": "A: Paris",
            "logprobs": [
                {"top_logprobs": [{"token": "A", "logprob": -0.1}]},
                {"top_logprobs": [{"token": ":", "logprob": -0.2}]}
            ],
        }
        f.write(json.dumps(record))

    # 2. Create dummy config
    output_dir = tmp_path / "artifacts/kd_model"
    final_dir = output_dir / "final"
    config = {
        "student_model": "sshleifer/tiny-gpt2",
        "teacher_model": "sshleifer/tiny-gpt2",
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
        "kd_alpha": 0.5,
        "kd_temp": 2.0,
        "bf16": False
    }

    # 3. Run train_kd function directly
    train_kd.run_train_kd(config)

    # 4. Validate that the final model was saved
    assert final_dir.is_dir()
    assert (final_dir / "pytorch_model.bin").exists()
    assert (final_dir / "config.json").exists()
    assert (final_dir / "tokenizer.json").exists()

    # 5. Check for trainer state
    assert (output_dir / "trainer_state.json").exists()
