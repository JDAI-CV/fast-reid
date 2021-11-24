# encoding: utf-8
from fastreid.utils.registry import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")
EVALUATOR_REGISTRY.__doc__ = """
Registry for reid evaluation in a trainer.
"""

