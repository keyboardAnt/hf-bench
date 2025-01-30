import click
from pathlib import Path
from typing import Optional

class CommonOptions:
    """Shared CLI options across commands"""
    
    def config_option(f):
        return click.option(
            '--config', '-c',
            type=click.Path(exists=True, path_type=Path),
            help='Path to config file',
            required=True
        )(f)
    
    def verbose_option(f):
        return click.option(
            '--verbose', '-v',
            count=True,
            help='Increase verbosity (can be used multiple times)'
        )(f)

def validate_config(ctx, param, value: Optional[Path]):
    """Validate and load config file"""
    if value is None:
        return None
    try:
        from hf_bench.utils.config import load_config
        return load_config(value)
    except Exception as e:
        raise click.BadParameter(f"Invalid config file: {e}") 