from src.data import DatasetManager
from src.config import Cfg
from datasets import DatasetDict

def test_dataset_manager_load_tokenized(cfg: Cfg, mock_dataset):
    """DatasetManager가 모의 데이터셋을 올바르게 토큰화하는지 검증"""
    dm = DatasetManager(cfg)
    tokenized_ds = dm.load_tokenized()

    assert isinstance(tokenized_ds, DatasetDict)
    assert "train" in tokenized_ds
    assert "validation" in tokenized_ds

    # 반환된 데이터셋의 컬럼과 shape 확인
    sample = tokenized_ds["train"][0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert cfg.dataset.label_col in sample
    assert len(sample["input_ids"]) == cfg.dataset.max_length

    # mock_dataset이 호출되었는지 확인
    mock_dataset.assert_called_once_with(cfg.dataset.name, cfg.dataset.subset)
