import click
from pathlib import Path

from hf_bench.runners import get_runner
from .base import CommonOptions, validate_config

@click.group()
def benchmark_cli():
    """Benchmark execution and analysis commands"""
    pass

@benchmark_cli.command()
@CommonOptions.config_option
@CommonOptions.verbose_option
@click.pass_context
def run(ctx, config: Path, verbose: int):
    """Run benchmark job"""
    config = validate_config(ctx, None, config)
    
    # Get runner
    from hf_bench.runners import get_runner
    runner = get_runner(config.cluster_config)
    
    # Submit job
    job_id = runner.submit('benchmark.py')
    click.echo(f"Submitted job {job_id}")

@benchmark_cli.command()
@CommonOptions.config_option
@click.argument('job_id')
@click.pass_context
def status(ctx, config: Path, job_id: str):
    """Get status of benchmark job"""
    config = validate_config(ctx, None, config)
    runner = get_runner(config.cluster_config)
    status = runner.get_status(job_id)
    click.echo(f"Job {job_id} status: {status}") 