"""Predictors. Importing this package registers each one."""

from predictor.baseline import BaselinePredictor
from predictor.predictor import BasePredictor

__all__ = ["BasePredictor", "BaselinePredictor"]
