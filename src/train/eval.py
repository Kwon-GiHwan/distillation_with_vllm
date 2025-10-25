import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import Cfg


class Evaluator:
    def __init__(self, cfg: Cfg):
     self.cfg = cfg


    def accuracy(self, model_dir: str) -> float:
        ds = load_dataset(self.cfg.dataset.name, self.cfg.dataset.subset)
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()


        val = ds["validation"]
        texts = val[self.cfg.dataset.text_col]
        labels = val[self.cfg.dataset.label_col]


        acc_cnt = 0
        with torch.no_grad():
            for i in range(0, len(texts), self.cfg.train.bsz_eval):
                chunk = texts[i:i+self.cfg.train.bsz_eval]
                enc = tok(chunk, padding=True, truncation=True, max_length=self.cfg.dataset.max_length, return_tensors="pt").to(device)
                logits = model(**enc).logits
                preds = logits.argmax(-1).cpu().tolist()
                acc_cnt += sum(int(p == y) for p, y in zip(preds, labels[i:i+self.cfg.train.bsz_eval]))
        return acc_cnt / len(val)


    def compare(self) -> dict:
        res = {}
        sft_dir = os.path.join(self.cfg.train.output_dir_sft, "best")
        kd_dir = os.path.join(self.cfg.train.output_dir_kd, "best")
        if os.path.isdir(sft_dir):
            res["SFT_acc"] = self.accuracy(sft_dir)
        if os.path.isdir(kd_dir):
            res["KD_acc"] = self.accuracy(kd_dir)
        return res