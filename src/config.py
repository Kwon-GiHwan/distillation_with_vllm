from dataclasses import dataclass
from typing import Dict, List
import yaml


@dataclass
class DatasetCfg:
    name: str
    subset: str
    text_col: str
    label_col: str
    max_length: int


@dataclass
class TeacherCfg:
    api_base: str
    api_key: str
    model: str
    temperature: float
    verbalizers: Dict[str, List[str]]
    system_prompt: str


@dataclass
class StudentCfg:
    model_id: str


@dataclass
class TrainCfg:
    output_dir_sft: str
    output_dir_kd: str
    lr: float
    epochs: int
    weight_decay: float
    bsz_train: int
    bsz_eval: int
    logging_steps: int


@dataclass
class KDCfg:
    alpha: float
    temperature: float


@dataclass
class CacheCfg:
    kd_dataset: str


@dataclass
class Cfg:
    seed: int
    dataset: DatasetCfg
    teacher: TeacherCfg
    student: StudentCfg
    train: TrainCfg
    kd: KDCfg
    cache: CacheCfg


class ConfigLoader:
    @staticmethod
    def load(path: str) -> Cfg:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Cfg(
        seed=raw["seed"],
        dataset=DatasetCfg(**raw["dataset"]),
        teacher=TeacherCfg(**raw["teacher"]),
        student=StudentCfg(**raw["student"]),
        train=TrainCfg(**raw["train"]),
        kd=KDCfg(**raw["kd"]),
        cache=CacheCfg(**raw["cache"]),
    )