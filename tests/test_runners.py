import pytest
import time
from pathlib import Path
from datetime import datetime
from hf_bench.runners import get_runner
from hf_bench.runners.lsf import LSFRunner
from hf_bench.schemas import (
    ClusterConfig, LSFConfig, LSFHardwareRequest, LSFEnvironment,
    GPUConfig, HardwareResponse, LoggingConfig
)

@pytest.fixture
def minimal_lsf_config(tmp_path_factory):
    """Create a minimal LSF configuration for testing."""
    # Create directory in a shared location instead of /tmp
    shared_tmp = Path("/home/projects/dharel/nadavt/tmp/hf-bench-tests")
    shared_tmp.mkdir(parents=True, exist_ok=True)
    test_dir = shared_tmp / str(time.time())
    test_dir.mkdir(parents=True)
    
    return ClusterConfig(
        type="lsf",
        lsf=LSFConfig(
            hardware_request=LSFHardwareRequest(
                gpu=GPUConfig(
                    type=None,
                    count=1,
                    memory_gb=80,
                    is_exclusive=True
                ),
                cpu_cores=8,
                memory_gb=200,
                queue_names=["long-gpu"],
                num_hosts=1,
                num_processes=1
            ),
            environment=LSFEnvironment(
                modules=["CUDA/12.4.0"]
            )
        ),
        slurm=None,
        logging=LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            output_dir=test_dir,
            filename_pattern="{timestamp}_jobid_{job_id}_benchmark.log"
        )
    )

def test_lsf_runner_creation(minimal_lsf_config):
    """Test that we can create an LSF runner."""
    runner = get_runner(minimal_lsf_config)
    assert isinstance(runner, LSFRunner)

def test_bsub_command_generation(minimal_lsf_config):
    """Test that the LSF runner generates correct bsub commands."""
    runner = LSFRunner(minimal_lsf_config)
    script_path = Path("test_script.py")
    cmd = runner.get_submit_command(script_path)
    
    cmd_str = " ".join(cmd)
    assert "bsub" in cmd_str
    assert "-gpu" in cmd_str
    assert "num=1:j_exclusive=yes:gmem=80GB" in cmd_str
    assert "rusage[mem=200GB]" in cmd_str
    assert "affinity[core(8)]" in cmd_str
    assert "CUDA/12.4.0" in cmd_str
