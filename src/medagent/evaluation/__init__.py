"""MedAgent 评测模块。"""

from .dataset import load_eval_dataset
from .runner import run_evaluation

__all__ = ["load_eval_dataset", "run_evaluation"]
