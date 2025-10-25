import os
import math
import time
from typing import Dict, List
import requests
from datasets import load_dataset, DatasetDict
from src.config import Cfg


class TeacherClientVLLM:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.url = f"{cfg.teacher.api_base}/chat/completions"
        self.headers = {"Authorization": f"Bearer {cfg.teacher.api_key}"}


    def _softmax_from_logprobs(self, label_logprob_map: Dict[str, float], temperature: float) -> List[float]:
        xs = {k: v / temperature for k, v in label_logprob_map.items()}
        m = max(xs.values())
        exps = {k: math.exp(v - m) for k, v in xs.items()}
        z = sum(exps.values())
        neg = exps.get("negative", 0.0) / z if z > 0 else 0.5
        pos = exps.get("positive", 0.0) / z if z > 0 else 0.5
        return [neg, pos]


    def _extract_label_logprobs(self, top_logprobs) -> Dict[str, float]:
        cand = {"positive": float("-inf"), "negative": float("-inf")}
        if not top_logprobs:
          return cand
        for item in top_logprobs:
            tok = item.get("token", "")
            lp = item.get("logprob", None)
            if lp is None:
                continue
            for lab, variants in self.cfg.teacher.verbalizers.items():
                if tok in variants:
                    cand[lab] = max(cand[lab], lp)
        return cand


    def classify_batch(self, texts: List[str]) -> List[List[float]]:
        logits = []
        for s in texts:
            payload = {
                "model": self.cfg.teacher.model,
                "messages": [
                {"role": "system", "content": self.cfg.teacher.system_prompt},
                {"role": "user", "content": s},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10,
                }
            r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            label_lp = self._extract_label_logprobs(top)


            out_token = data["choices"][0]["message"]["content"].strip().lower()
            if out_token in ("positive", "negative") and not any(v > -1e9 for v in label_lp.values()):
                label_lp[out_token] = 0.0
                other = "negative" if out_token == "positive" else "positive"
                label_lp[other] = float("-inf")

            neg_p, pos_p = self._softmax_from_logprobs(label_lp, self.cfg.teacher.temperature)
            logits.append([math.log(max(neg_p, 1e-40)), math.log(max(pos_p, 1e-40))])
            time.sleep(0.005)
        return logits


    def collect(self) -> str:
        ds = load_dataset(self.cfg.dataset.name, self.cfg.dataset.subset)
        text_col = self.cfg.dataset.text_col
        for split in ("train", "validation"):
            texts = ds[split][text_col]
            all_logits = []
            B = 64
            for i in range(0, len(texts), B):
                all_logits.extend(self.classify_batch(texts[i:i+B]))
            ds[split] = ds[split].add_column("teacher_logits", all_logits)
        out = DatasetDict({"train": ds["train"], "validation": ds["validation"]})
        os.makedirs(os.path.dirname(self.cfg.cache.kd_dataset), exist_ok=True)
        out.save_to_disk(self.cfg.cache.kd_dataset)
        return self.cfg.cache.kd_dataset