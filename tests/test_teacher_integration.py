import os
from unittest.mock import patch
from src.config import Cfg
from src.train.teacher import TeacherClientVLLM

def test_teacher_collect_smoke(cfg: Cfg, mock_dataset, mock_vllm_post, monkeypatch):
    """
    (통합/스모크) TeacherClientVLLM.collect가 모의 API 호출 후
    올바른 경로로 save_to_disk를 호출하는지 검증
    """
    # DatasetDict.save_to_disk를 모의 처리
    mock_save = patch("datasets.DatasetDict.save_to_disk").start()

    teacher = TeacherClientVLLM(cfg)
    cache_path = teacher.collect()

    # 캐시 경로가 설정과 일치하는지 확인
    assert cache_path == cfg.cache.kd_dataset

    # save_to_disk가 올바른 경로로 호출되었는지 확인
    mock_save.assert_called_once_with(cfg.cache.kd_dataset)

    # mock_vllm_post가 호출되었는지 확인
    assert mock_vllm_post.call_count == 4 # train/validation 각 2개 샘플

    # patch 중지
    patch.stopall()
