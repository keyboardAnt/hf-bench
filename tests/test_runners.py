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

@pytest.mark.integration
def test_simple_job_submission(minimal_lsf_config):
    """Test submitting a simple job to LSF."""
    # Create a simple test script in the same shared directory as logs
    script_path = Path(minimal_lsf_config.logging.output_dir) / "test_job.py"
    script_content = """
import time
print("Starting test job")
time.sleep(10)
print("Test job completed")
"""
    script_path.write_text(script_content)
    
    # Submit job
    runner = LSFRunner(minimal_lsf_config)
    job_id = runner.submit(script_path)
    
    # Basic checks
    assert job_id is not None
    assert isinstance(job_id, str)
    
    # Get hardware info
    hw_info = runner.get_hardware_info(job_id)
    assert isinstance(hw_info, HardwareResponse)
    assert hw_info.job_id == job_id
    assert isinstance(hw_info.submission_time, datetime)
    
    # Wait for job to start (with timeout)
    max_wait = 60  # seconds
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < max_wait:
        status = runner.get_status(job_id)
        if status == "RUN":
            break
        time.sleep(5)
    
    # Check running job info
    hw_info = runner.get_hardware_info(job_id)
    # Only check that we got back a HardwareResponse object
    assert isinstance(hw_info, HardwareResponse)
    assert hw_info.job_id == job_id  # This is required
    
    # Wait for completion
    while runner.get_status(job_id) != "DONE":
        time.sleep(5)
    
    # Check final hardware info
    hw_info = runner.get_hardware_info(job_id)
    assert isinstance(hw_info, HardwareResponse)

@pytest.mark.integration
def test_gpu_job_submission(minimal_lsf_config):
    """Test submitting a job that uses GPU."""
    # Create script in the shared directory
    script_path = Path(minimal_lsf_config.logging.output_dir) / "gpu_test.py"
    script_content = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name()}")
"""
    script_path.write_text(script_content)
    
    runner = LSFRunner(minimal_lsf_config)
    job_id = runner.submit(script_path)
    
    # Wait for completion
    max_wait = 120  # seconds
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < max_wait:
        status = runner.get_status(job_id)
        if status == "DONE":
            break
        time.sleep(5)
    
    hw_info = runner.get_hardware_info(job_id)
    # Only assert on the required fields
    assert isinstance(hw_info, HardwareResponse)
    assert hw_info.job_id == job_id
    assert hw_info.loaded_modules == ["CUDA/12.4.0"]