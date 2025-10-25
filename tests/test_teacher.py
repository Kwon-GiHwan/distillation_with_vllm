import math
import pytest
from src.config import Cfg
from src.train.teacher import TeacherClientVLLM

@pytest.fixture
def teacher_client(cfg: Cfg) -> TeacherClientVLLM:
    """테스트용 TeacherClientVLLM 객체를 제공하는 Fixture"""
    # Import 경로가 src/train/teacher.py 이므로 conftest.py의 monkeypatch가 동작하지 않을 수 있음
    # 테스트 파일 내에서 직접 경로를 지정하여 TeacherClientVLLM 객체 생성
    from src.train.teacher import TeacherClientVLLM
    return TeacherClientVLLM(cfg)

def test_extract_label_logprobs_normal(teacher_client: TeacherClientVLLM):
    """정상적인 top_logprobs 입력에 대한 파싱 테스트"""
    top_logprobs = [
        {"token": "positive", "logprob": -0.1},
        {"token": "negative", "logprob": -2.5},
        {"token": "other", "logprob": -5.0},
    ]
    result = teacher_client._extract_label_logprobs(top_logprobs)
    assert result["positive"] == -0.1
    assert result["negative"] == -2.5

def test_extract_label_logprobs_empty(teacher_client: TeacherClientVLLM):
    """빈 top_logprobs 입력 처리 테스트"""
    result = teacher_client._extract_label_logprobs([])
    assert result["positive"] == float("-inf")
    assert result["negative"] == float("-inf")

def test_extract_label_logprobs_missing_key(teacher_client: TeacherClientVLLM):
    """'logprob' 키가 누락된 경우의 처리 테스트"""
    top_logprobs = [{"token": "positive"}]
    result = teacher_client._extract_label_logprobs(top_logprobs)
    assert result["positive"] == float("-inf")

def test_softmax_from_logprobs(teacher_client: TeacherClientVLLM):
    """소프트맥스 계산 로직 검증"""
    label_logprob_map = {"positive": -0.1, "negative": -2.5}
    # 온도(temperature)를 1.0으로 가정하고 테스트
    neg, pos = teacher_client._softmax_from_logprobs(label_logprob_map, temperature=1.0)

    # 직접 계산한 값과 비교
    expected_pos = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.5))
    expected_neg = 1.0 - expected_pos

    assert math.isclose(pos, expected_pos, rel_tol=1e-9)
    assert math.isclose(neg, expected_neg, rel_tol=1e-9)
    assert pos > neg

def test_softmax_with_temperature(teacher_client: TeacherClientVLLM):
    """온도(temperature)가 적용된 소프트맥스 계산 검증"""
    label_logprob_map = {"positive": -0.1, "negative": -2.5}
    temp = 2.0
    neg, pos = teacher_client._softmax_from_logprobs(label_logprob_map, temperature=temp)

    # 직접 계산한 값과 비교
    scaled_lp = {k: v / temp for k, v in label_logprob_map.items()}
    expected_pos = math.exp(scaled_lp["positive"]) / (math.exp(scaled_lp["positive"]) + math.exp(scaled_lp["negative"]))
    expected_neg = 1.0 - expected_pos

    assert math.isclose(pos, expected_pos, rel_tol=1e-9)
    assert math.isclose(neg, expected_neg, rel_tol=1e-9)
