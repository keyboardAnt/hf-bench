import subprocess
import re
from datetime import datetime
from pathlib import Path
from .base import BaseRunner
from hf_bench.schemas import HardwareResponse

class LSFRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.lsf_config = config.lsf
        self._hardware_info: dict[str, HardwareResponse] = {}

    def submit(self, script_path: str, *args):
        """Submit a job to LSF and collect initial hardware information."""
        cmd = self._build_bsub_command(script_path, *args)
        bsub_command = " ".join(cmd)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Extract job ID from bsub output
            job_id = result.stdout.split("<")[1].split(">")[0]
            
            # Initialize hardware response with submission info
            self._hardware_info[job_id] = HardwareResponse(
                job_id=job_id,
                queue_name=self._get_queue_name(job_id),
                submission_time=datetime.now(),
                loaded_modules=self.lsf_config.environment.modules,
                bsub_command=bsub_command
            )
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Job submission failed: {e.stderr}")

    def get_hardware_info(self, job_id: str) -> HardwareResponse:
        """Get hardware information for a specific job."""
        if job_id not in self._hardware_info:
            raise KeyError(f"No hardware information found for job {job_id}")
            
        # Update hardware info if job is running or completed
        self._update_hardware_info(job_id)
        return self._hardware_info[job_id]

    def _update_hardware_info(self, job_id: str):
        """Update hardware information using bjobs and other LSF commands."""
        status = self.get_status(job_id)
        hw_info = self._hardware_info[job_id]
        
        if status in ["RUN", "DONE", "EXIT"]:
            # Get detailed job info using bjobs -l
            cmd = ["bjobs", "-l", job_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if status == "RUN" and not hw_info.start_time:
                # Update start time and node information
                hw_info.start_time = self._parse_start_time(result.stdout)
                hw_info.node_name = self._parse_node_name(result.stdout)
                
                # Get node-specific information
                self._update_node_info(job_id)
                
            if status in ["DONE", "EXIT"]:
                # Update resource usage information
                self._update_resource_usage(job_id)

    def _update_node_info(self, job_id: str):
        """Update node-specific information using LSF commands."""
        hw_info = self._hardware_info[job_id]
        
        if not hw_info.node_name:
            return
            
        # Get host architecture and OS info
        cmd = ["lshosts", hw_info.node_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        hw_info.host_architecture = self._parse_architecture(result.stdout)
        hw_info.os_info = self._parse_os_info(result.stdout)
        
        # Get GPU information using nvidia-smi
        self._update_gpu_info(job_id)

    def _update_gpu_info(self, job_id: str):
        """Update GPU information using nvidia-smi."""
        hw_info = self._hardware_info[job_id]
        
        # Run nvidia-smi through bsub
        cmd = [
            "bsub", "-Is", "nvidia-smi", "--query-gpu=gpu_name,driver_version,cuda_version",
            "--format=csv,noheader"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            gpu_info = result.stdout.strip().split(",")
            
            hw_info.gpu_type = gpu_info[0].strip()
            hw_info.gpu_driver_version = gpu_info[1].strip()
            hw_info.cuda_version = gpu_info[2].strip()
            
            # Get GPU interconnect information
            hw_info.network = self._get_gpu_interconnect()
            
        except subprocess.CalledProcessError:
            pass  # Handle gracefully if nvidia-smi fails

    def _update_resource_usage(self, job_id: str):
        """Update resource usage information for completed jobs."""
        cmd = ["bacct", "-l", job_id]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            hw_info = self._hardware_info[job_id]
            
            # Parse bacct output to populate:
            # - max_memory_gb
            # - max_gpu_memory_gb
            # - cpu_time
            # - wall_time
            # - avg_cpu_util
            self._parse_resource_usage(result.stdout, hw_info)
            
        except subprocess.CalledProcessError:
            pass  # Handle gracefully if bacct fails

    # Various helper methods for parsing LSF output
    def _parse_start_time(self, output: str) -> datetime:
        """Parse job start time from bjobs output."""
        # Implementation details...
        pass

    def _parse_node_name(self, output: str) -> str:
        """Parse node name from bjobs output."""
        # Implementation details...
        pass

    def _get_queue_name(self, job_id: str) -> str:
        """Get the actual queue name assigned to the job."""
        # Implementation details...
        pass

    def _get_gpu_interconnect(self) -> str:
        """Determine GPU interconnect type (PCIe/NVLink)."""
        # Implementation details...
        pass

    def _parse_resource_usage(self, output: str, hw_info: HardwareResponse):
        """Parse resource usage information from bacct output."""
        # Implementation details...
        pass

    def _build_bsub_command(self, script_path: str, *args) -> list:
        """Build the bsub command with all necessary arguments."""
        hw = self.lsf_config.hardware_request
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(self.config.logging.output_dir) / f"{timestamp}_jobid_%J_benchmark.log"
        
        cmd = ["bsub"]
        
        # Queue settings
        cmd.extend(["-q", ",".join(hw.queue_names)])
        
        # GPU settings
        gpu_spec = f"num={hw.gpu.count}"
        if hw.gpu.is_exclusive:
            gpu_spec += ":j_exclusive=yes"
        if hw.gpu.memory_gb:
            gpu_spec += f":gmem={hw.gpu.memory_gb}GB"
        cmd.extend(["-gpu", f'"{gpu_spec}"'])
        
        # CPU and memory settings
        cmd.extend([
            "-R", f"rusage[mem={hw.memory_gb}GB]",
            "-R", f"affinity[core({hw.cpu_cores})]",
            "-R", f"span[hosts={hw.num_hosts}]",
            "-n", str(hw.num_processes),
            "-M", f"{hw.memory_gb}GB",
            "-o", str(log_path)
        ])
        
        # Build the execution command
        exec_cmd = []
        
        # Add module loads
        for module in self.lsf_config.environment.modules:
            exec_cmd.extend(["module load", module, "&&"])
        
        # Add the script and its arguments
        exec_cmd.extend(["python", str(script_path)])
        exec_cmd.extend(str(arg) for arg in args)
        
        # Join the execution command
        cmd.append(" ".join(exec_cmd))
        
        return cmd

    def get_status(self, job_id: str) -> str:
        """Get the status of an LSF job."""
        cmd = ["bjobs", "-noheader", job_id]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            # Parse the status from bjobs output
            if not result.stdout.strip():
                return "UNKNOWN"
            return result.stdout.split()[2]  # Status is the third column in bjobs output
        except subprocess.CalledProcessError:
            return "ERROR" 