import pytest
import yaml
from unittest.mock import MagicMock, patch
from datasets import Dataset, DatasetDict
from src.config import ConfigLoader

@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """테스트용 Cfg 객체를 제공하는 Fixture"""
    tmp_dir = tmp_path_factory.mktemp("config")
    config_path = tmp_dir / "config.yaml"
    config_content = {
        "seed": 42,
        "dataset": {"name": "glue", "subset": "sst2", "text_col": "sentence", "label_col": "label", "max_length": 128},
        "teacher": {
            "api_base": "http://localhost:8000/v1",
            "api_key": "dummy",
            "model": "mock-teacher",
            "temperature": 2.0,
            "verbalizers": {"positive": ["positive"], "negative": ["negative"]},
            "system_prompt": "Classify: positive or negative.",
        },
        "student": {"model_id": "prajjwal1/bert-tiny"},
        "train": {
            "output_dir_sft": str(tmp_dir / "out_sft"),
            "output_dir_kd": str(tmp_dir / "out_kd"),
            "lr": 1e-4,
            "epochs": 1,
            "weight_decay": 0.01,
            "bsz_train": 2,
            "bsz_eval": 2,
            "logging_steps": 1,
        },
        "kd": {"alpha": 0.5, "temperature": 2.0},
        "cache": {"kd_dataset": str(tmp_dir / "cache/kd_data")},
    }
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    return ConfigLoader.load(str(config_path))

@pytest.fixture
def mock_dataset(monkeypatch):
    """datasets.load_dataset을 작은 더미 데이터로 모킹"""
    dummy_data = {"sentence": ["a positive example", "a negative example"], "label": [1, 0]}
    dummy_ds = Dataset.from_dict(dummy_data)
    dummy_dict = DatasetDict({"train": dummy_ds, "validation": dummy_ds})

    mock_load = MagicMock(return_value=dummy_dict)

    # Patch all known locations where load_dataset is imported
    targets = [
        "src.data.load_dataset",
        "src.train.teacher.load_dataset",
        "src.train.eval.load_dataset",
    ]

    for target in targets:
        monkeypatch.setattr(target, mock_load, raising=False)

    return mock_load

@pytest.fixture
def mock_vllm_post(monkeypatch):
    """requests.post를 vLLM 응답 형식으로 모킹"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    # Positive 예시 응답
    positive_response = {
      "choices": [{
        "message": {"content": "positive"},
        "logprobs": {"content": [{
          "top_logprobs": [
            {"token": "positive", "logprob": -0.1},
            {"token": "negative", "logprob": -2.5}
          ]
        }]}
      }]
    }

    # Negative 예시 응답
    negative_response = {
      "choices": [{
        "message": {"content": "negative"},
        "logprobs": {"content": [{
          "top_logprobs": [
            {"token": "positive", "logprob": -2.5},
            {"token": "negative", "logprob": -0.1}
          ]
        }]}
      }]
    }

    # 요청 내용에 따라 다른 응답 반환
    def mock_post_logic(*args, **kwargs):
        json_payload = kwargs.get("json", {})
        user_content = ""
        if "messages" in json_payload:
            for msg in json_payload["messages"]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break

        if "negative" in user_content:
            mock_response.json.return_value = negative_response
        else:
            mock_response.json.return_value = positive_response

        return mock_response

    mock_post = MagicMock(side_effect=mock_post_logic)
    monkeypatch.setattr("requests.post", mock_post)
    return mock_post
