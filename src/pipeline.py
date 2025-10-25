from .config import Cfg
from src.train.teacher import TeacherClientVLLM
from src.train.trainer import TrainerEngine
from src.train.eval import Evaluator


class Pipeline:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.teacher = TeacherClientVLLM(cfg)
        self.engine = TrainerEngine(cfg)
        self.evaluator = Evaluator(cfg)


    def sft(self):
        self.engine.train(mode="sft")


    def kd_collect(self):
        return self.teacher.collect()


    def kd_train(self):
        self.engine.train(mode="kd")


    def eval_all(self):
        return self.evaluator.compare()


    def run_all(self):
        self.sft()
        self.kd_collect()
        self.kd_train()
        return self.eval_all()