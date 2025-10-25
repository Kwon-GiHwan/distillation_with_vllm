import os
from unittest.mock import patch
from src.config import Cfg
from src.train.eval import Evaluator

@patch("src.train.eval.Evaluator.accuracy")
def test_evaluator_compare_smoke(mock_accuracy, cfg: Cfg):
    """
    (통합/스모크) Evaluator.compare가 더미 모델 디렉터리가 있을 때
    결과 딕셔너리를 올바르게 반환하는지 테스트
    """
    # accuracy가 호출될 때마다 다른 값을 반환하도록 설정
    mock_accuracy.side_effect = [0.85, 0.95]

    # 더미 모델 디렉터리 생성
    sft_dir = os.path.join(cfg.train.output_dir_sft, "best")
    kd_dir = os.path.join(cfg.train.output_dir_kd, "best")
    os.makedirs(sft_dir)
    os.makedirs(kd_dir)

    evaluator = Evaluator(cfg)
    results = evaluator.compare()

    # 반환된 결과 딕셔너리 검증
    assert "SFT_acc" in results
    assert "KD_acc" in results
    assert results["SFT_acc"] == 0.85
    assert results["KD_acc"] == 0.95

    # accuracy가 올바른 경로로 호출되었는지 확인
    mock_accuracy.assert_any_call(sft_dir)
    mock_accuracy.assert_any_call(kd_dir)
    assert mock_accuracy.call_count == 2
