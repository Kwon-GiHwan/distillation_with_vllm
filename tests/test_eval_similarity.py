"""Tests for `app.eval.eval_similarity`."""
from typer.testing import CliRunner
import yaml
import orjson
import torch
import pytest
from unittest.mock import MagicMock
from app.eval import eval_similarity

from app.cli import app

runner = CliRunner()

def test_run_eval_similarity(mocker, tmp_path):
    """Verify `run_eval_similarity` prints correct F1, P, and R scores."""
    # 1. Mock the generate_batch function
    mock_generate = mocker.patch("app.eval.eval_similarity.generate_batch")
    mock_generate.side_effect = [
        ["teacher response 1", "teacher response 2"],
        ["student response 1", "student response 2"]
    ]

    # 2. Mock BERTScore
    mock_bert_score = mocker.patch("app.eval.eval_similarity.bert_score")
    mock_bert_score.return_value = (
        torch.tensor([0.9, 0.8]),  # Precision
        torch.tensor([0.85, 0.75]), # Recall
        torch.tensor([0.87, 0.77]), # F1
    )

    # 3. Mock dataset loading
    mock_dataset = MagicMock()
    mock_dataset.select.return_value = [
        {"instruction": "q1", "input": ""},
        {"instruction": "q2", "input": ""}
    ]
    mocker.patch("app.eval.eval_similarity.load_dataset", return_value=mock_dataset)

    # 4. Mock model and tokenizer loading
    mocker.patch("app.eval.eval_similarity.AutoModelForCausalLM.from_pretrained")
    mocker.patch("app.eval.eval_similarity.AutoTokenizer.from_pretrained")

    # 5. Create a dummy config
    config = {
        "teacher_path": "dummy-teacher",
        "student_path": "dummy-student",
        "sample_size": 2
    }

    # 6. Run the eval-similarity function directly
    results = eval_similarity.run_eval_similarity(**config)

    # 7. Check the returned results
    assert results["f1"] == pytest.approx(0.82, 1e-4)
    assert results["precision"] == pytest.approx(0.85, 1e-4)
    assert results["recall"] == pytest.approx(0.80, 1e-4)
