import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, default_data_collator
from typing import Dict
from src.config import Cfg
from src.data import DatasetManager

from packaging import version
import transformers




class KDTrainer(Trainer):
    def __init__(self, *args, kd_alpha: float = 0.5, kd_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")
        teacher_logits = inputs.pop("teacher_logits", None)
        outputs = model(**inputs)
        student_logits = outputs.logits
        ce_loss = F.cross_entropy(student_logits, labels)
        if teacher_logits is not None:
            t = self.kd_temperature
            teacher_logits = teacher_logits.to(student_logits.device)
            log_p_s = F.log_softmax(student_logits / t, dim=-1)
            p_t = F.softmax(teacher_logits / t, dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)
            loss = self.kd_alpha * ce_loss + (1 - self.kd_alpha) * kl
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


class TrainerEngine:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.dm = DatasetManager(cfg)


    def _compute_metrics(self, eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds)}


    def train(self, mode: str = "sft"):
        self.dm.set_seed(self.cfg.seed)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.student.model_id)
        if mode == "kd":
            ds = load_from_disk(self.cfg.cache.kd_dataset)
            out_dir = self.cfg.train.output_dir_kd
            trainer_cls = KDTrainer
            trainer_kwargs = dict(kd_alpha=self.cfg.kd.alpha, kd_temperature=self.cfg.kd.temperature)
        else:
            ds = self.dm.load_tokenized()
            out_dir = self.cfg.train.output_dir_sft
            trainer_cls = Trainer
            trainer_kwargs = {}


        model = AutoModelForSequenceClassification.from_pretrained(self.cfg.student.model_id, num_labels=2)


        args = TrainingArguments(
            output_dir=out_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.cfg.train.lr,
            per_device_train_batch_size=self.cfg.train.bsz_train,
            per_device_eval_batch_size=self.cfg.train.bsz_eval,
            num_train_epochs=self.cfg.train.epochs,
            weight_decay=self.cfg.train.weight_decay,
            logging_steps=self.cfg.train.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
        )


        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics,
            **trainer_kwargs,
        )
        trainer.train()
        trainer.save_model(f"{out_dir}/best")