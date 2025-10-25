from src.config import Cfg, DatasetCfg

def test_config_loader(cfg: Cfg):
    """ConfigLoader가 YAML을 Cfg 데이터클래스로 올바르게 파싱하는지 검증"""
    assert isinstance(cfg, Cfg)
    assert isinstance(cfg.dataset, DatasetCfg)
    assert cfg.seed == 42
    assert cfg.student.model_id == "prajjwal1/bert-tiny"
    assert "positive" in cfg.teacher.verbalizers
