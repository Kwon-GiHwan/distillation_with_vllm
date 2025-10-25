from unittest.mock import patch, MagicMock
from src.config import Cfg
from src.pipeline import Pipeline

@patch("src.pipeline.Evaluator")
@patch("src.pipeline.TrainerEngine")
@patch("src.pipeline.TeacherClientVLLM")
def test_pipeline_run_all_smoke(mock_teacher_client, mock_trainer_engine, mock_evaluator, cfg: Cfg):
    """
    (통합/스모크) Pipeline.run_all이 전체 파이프라인 흐름을
    올바른 순서로 호출하는지 검증
    """
    # 각 컴포넌트의 메서드가 호출되었는지 추적하기 위한 모의 객체 설정
    mock_teacher_instance = mock_teacher_client.return_value
    mock_trainer_instance = mock_trainer_engine.return_value
    mock_evaluator_instance = mock_evaluator.return_value

    # eval_all이 최종 결과를 반환하도록 설정
    mock_evaluator_instance.compare.return_value = {"SFT_acc": 0.8, "KD_acc": 0.9}

    pipeline = Pipeline(cfg)
    result = pipeline.run_all()

    # 각 메서드가 한 번씩 호출되었는지 확인
    mock_trainer_instance.train.assert_any_call(mode="sft")
    mock_teacher_instance.collect.assert_called_once()
    mock_trainer_instance.train.assert_any_call(mode="kd")
    mock_evaluator_instance.compare.assert_called_once()

    # train 메서드는 총 두 번 호출되어야 함 (sft, kd)
    assert mock_trainer_instance.train.call_count == 2

    # 최종 결과가 eval_all의 반환값과 일치하는지 확인
    assert "SFT_acc" in result
    assert result["KD_acc"] == 0.9
