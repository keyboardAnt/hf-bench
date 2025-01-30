import click
from pathlib import Path
from datetime import datetime
from .base import CommonOptions, validate_config

@click.group()
def cluster_cli():
    """Cluster management commands"""
    pass

@cluster_cli.command()
@CommonOptions.config_option
@click.option('--dry-run', is_flag=True, help='Print command without executing')
@click.pass_context
def get_command(ctx, config: Path, dry_run: bool):
    """Generate cluster submission command"""
    config = validate_config(ctx, None, config)
    
    # Get runner based on config
    from hf_bench.runners import get_runner
    runner = get_runner(config.cluster_config)
    
    # Generate command string
    cmd = runner.get_submit_command()
    
    # Print or execute
    if dry_run:
        click.echo(cmd)
    else:
        click.echo(cmd)
        if click.confirm('Execute this command?'):
            import subprocess
            subprocess.run(cmd, shell=True) 