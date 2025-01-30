import click
from .cli import cluster_cli, benchmark_cli

@click.group()
def main():
    """HF Bench CLI"""
    pass

main.add_command(cluster_cli, name='cluster')
main.add_command(benchmark_cli, name='benchmark')

if __name__ == '__main__':
    main() 