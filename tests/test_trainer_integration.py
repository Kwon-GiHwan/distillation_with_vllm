import os
import logging
import torch
from unittest.mock import patch, MagicMock
from src.config import Cfg
from src.train.trainer import TrainerEngine
from datasets import DatasetDict

@patch("src.train.trainer.TrainingArguments")
@patch("src.train.trainer.AutoModelForSequenceClassification.from_pretrained")
def test_trainer_engine_sft_smoke(mock_from_pretrained, mock_training_args, cfg: Cfg, mock_dataset):
    """(통합/스모크) TrainerEngine.train(mode='sft')가 save_model을 올바르게 호출하는지 검증"""
    args_instance = mock_training_args.return_value
    args_instance.batch_eval_metrics = False
    args_instance.seed = cfg.seed
    args_instance.full_determinism = False
    args_instance.deepspeed_plugin = None
    args_instance.parallelism_config = None
    args_instance.accelerator_config.gradient_accumulation_kwargs = None
    args_instance.get_process_log_level.return_value = logging.INFO
    args_instance.use_liger_kernel = False
    args_instance.eval_strategy = "no"
    args_instance.save_strategy = "no"
    args_instance.load_best_model_at_end = False
    args_instance.max_steps = -1
    args_instance.num_train_epochs = 1
    args_instance.label_smoothing_factor = 0.0
    args_instance.place_model_on_device = False
    args_instance.fp16_full_eval = False
    args_instance.bf16_full_eval = False
    args_instance.report_to = []
    args_instance.fsdp_config = {"xla_fsdp_v2": False, "xla": False}
    args_instance.label_names = None

    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.tp_size = None
    mock_model.is_parallelizable = False
    mock_model.hf_device_map = None
    mock_model._modules = {}
    mock_from_pretrained.return_value = mock_model

    mock_train = patch("transformers.Trainer.train").start()
    mock_save = patch("transformers.Trainer.save_model").start()

    engine = TrainerEngine(cfg)
    engine.train(mode="sft")

    mock_train.assert_called_once()
    expected_save_path = os.path.join(cfg.train.output_dir_sft, "best")
    mock_save.assert_called_once_with(expected_save_path)

    patch.stopall()

@patch("src.train.trainer.TrainingArguments")
@patch("src.train.trainer.AutoModelForSequenceClassification.from_pretrained")
@patch("src.train.trainer.load_from_disk")
def test_trainer_engine_kd_smoke(mock_load_disk, mock_from_pretrained, mock_training_args, cfg: Cfg, mock_dataset):
    """(통합/스모크) TrainerEngine.train(mode='kd')가 save_model을 올바르게 호출하는지 검증"""
    args_instance = mock_training_args.return_value
    args_instance.batch_eval_metrics = False
    args_instance.seed = cfg.seed
    args_instance.full_determinism = False
    args_instance.deepspeed_plugin = None
    args_instance.parallelism_config = None
    args_instance.accelerator_config.gradient_accumulation_kwargs = None
    args_instance.get_process_log_level.return_value = logging.INFO
    args_instance.use_liger_kernel = False
    args_instance.eval_strategy = "no"
    args_instance.save_strategy = "no"
    args_instance.load_best_model_at_end = False
    args_instance.max_steps = -1
    args_instance.num_train_epochs = 1
    args_instance.label_smoothing_factor = 0.0
    args_instance.place_model_on_device = False
    args_instance.fp16_full_eval = False
    args_instance.bf16_full_eval = False
    args_instance.report_to = []
    args_instance.fsdp_config = {"xla_fsdp_v2": False, "xla": False}
    args_instance.label_names = None

    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.tp_size = None
    mock_model.is_parallelizable = False
    mock_model.hf_device_map = None
    mock_model._modules = {}
    mock_from_pretrained.return_value = mock_model

    dummy_ds = mock_dataset.return_value
    dummy_ds["train"] = dummy_ds["train"].add_column("teacher_logits", [[0.1, 0.9]] * len(dummy_ds["train"]))
    dummy_ds["validation"] = dummy_ds["validation"].add_column("teacher_logits", [[0.1, 0.9]] * len(dummy_ds["validation"]))
    mock_load_disk.return_value = dummy_ds

    mock_train = patch("transformers.Trainer.train").start()
    mock_save = patch("transformers.Trainer.save_model").start()

    engine = TrainerEngine(cfg)
    engine.train(mode="kd")

    mock_load_disk.assert_called_once_with(cfg.cache.kd_dataset)
    mock_train.assert_called_once()
    expected_save_path = os.path.join(cfg.train.output_dir_kd, "best")
    mock_save.assert_called_once_with(expected_save_path)

    patch.stopall()
