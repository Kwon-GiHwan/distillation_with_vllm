import torch
import pytest
from unittest.mock import MagicMock, Mock
from src.config import Cfg
from src.train.trainer import KDTrainer

@pytest.fixture
def dummy_model_inputs():
    """KDTrainer.compute_loss 테스트를 위한 더미 입력 데이터 생성"""
    inputs = {
        "input_ids": torch.randint(0, 100, (2, 10)),
        "attention_mask": torch.ones(2, 10),
        "label": torch.tensor([0, 1]),
    }
    return inputs

@pytest.fixture
def dummy_model_outputs():
    """더미 모델 출력 (logits) 생성"""
    logits = torch.randn(2, 2)
    outputs = MagicMock()
    outputs.logits = logits
    return outputs

def test_kdtrainer_ce_loss_only(cfg: Cfg, dummy_model_inputs, dummy_model_outputs):
    """teacher_logits가 없을 때 표준 Cross-Entropy 손실만 계산되는지 검증"""
    model = MagicMock(return_value=dummy_model_outputs)

    mock_self = Mock()
    mock_self.kd_alpha = cfg.kd.alpha
    mock_self.kd_temperature = cfg.kd.temperature

    # .copy()를 사용하여 원본 딕셔너리가 수정되는 것을 방지
    loss = KDTrainer.compute_loss(mock_self, model, dummy_model_inputs.copy())

    expected_loss = torch.nn.functional.cross_entropy(
        dummy_model_outputs.logits,
        dummy_model_inputs["label"]
    )

    assert torch.isclose(loss, expected_loss)

def test_kdtrainer_kd_loss(cfg: Cfg, dummy_model_inputs, dummy_model_outputs):
    """teacher_logits가 있을 때 KD 손실(CE + KL)이 올바르게 계산되는지 검증"""
    model = MagicMock(return_value=dummy_model_outputs)

    mock_self = Mock()
    mock_self.kd_alpha = cfg.kd.alpha
    mock_self.kd_temperature = cfg.kd.temperature

    inputs_with_teacher = dummy_model_inputs.copy()
    # teacher_logits 텐서를 별도 변수에 저장
    teacher_logits_tensor = torch.randn(2, 2)
    inputs_with_teacher["teacher_logits"] = teacher_logits_tensor

    # compute_loss는 inputs_with_teacher 딕셔너리를 수정함
    loss = KDTrainer.compute_loss(mock_self, model, inputs_with_teacher)

    student_logits = dummy_model_outputs.logits
    labels = dummy_model_inputs["label"]
    # 수정되지 않은 원본 텐서를 사용하여 예상 손실 계산
    teacher_logits = teacher_logits_tensor

    ce_loss = torch.nn.functional.cross_entropy(student_logits, labels)

    t = cfg.kd.temperature
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / t, dim=-1),
        torch.nn.functional.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean"
    ) * (t * t)

    expected_loss = cfg.kd.alpha * ce_loss + (1 - cfg.kd.alpha) * kl_div

    assert torch.isclose(loss, expected_loss)
