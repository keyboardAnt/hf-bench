from .base import BaseRunner
from .lsf import LSFRunner

def get_runner(config):
    """Factory function to get the appropriate runner based on config."""
    if config.type != "lsf":
        raise ValueError(f"Unsupported runner type: {config.type}. Only 'lsf' is currently supported")
    return LSFRunner(config)

__all__ = ['BaseRunner', 'LSFRunner', 'get_runner']
