import random
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Any
from .config import Cfg


class DatasetManager:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg


    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def load_tokenized(self) -> DatasetDict:
        ds = load_dataset(self.cfg.dataset.name, self.cfg.dataset.subset)
        tok = AutoTokenizer.from_pretrained(self.cfg.student.model_id)

        def _tok(batch):
            enc = tok(
            batch[self.cfg.dataset.text_col],
            padding="max_length",
            truncation=True,
            max_length=self.cfg.dataset.max_length,
            )
            enc[self.cfg.dataset.label_col] = batch[self.cfg.dataset.label_col]
            return enc

        tokenized = ds.map(_tok, batched=True, remove_columns=ds["train"].column_names)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", self.cfg.dataset.label_col])
        return tokenized