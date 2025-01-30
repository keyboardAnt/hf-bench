import subprocess
import re
from datetime import datetime
from pathlib import Path
from .base import BaseRunner
from hf_bench.schemas import ClusterConfig, HardwareResponse, LoggingConfig
from typing import Optional

class LSFRunner(BaseRunner):
    def __init__(self, cluster_config: ClusterConfig, logging_config: Optional[LoggingConfig] = None):
        super().__init__(cluster_config)
        self.config = cluster_config  # Store the full config
        self.lsf_config = cluster_config.lsf
        self.logging_config = logging_config
        self._hardware_info: dict[str, HardwareResponse] = {}

    def submit(self, script_path: str, *args):
        """Submit a job to LSF and collect initial hardware information."""
        cmd = self.get_submit_command(script_path, *args)
        bsub_command = " ".join(cmd)
        
        try:
            result = subprocess.run(bsub_command, shell=True, check=True, capture_output=True, text=True)
            
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

    def get_submit_command(self, script_path: str | None = None, *args) -> list | str:
        """Build the bsub command with all necessary arguments."""
        hw = self.lsf_config.hardware_request
        
        # Default to current directory if no logging config is provided
        output_dir = (self.logging_config.output_dir 
                     if self.logging_config is not None 
                     else Path.cwd())
        
        # Get log filename pattern, replacing {job_id} with LSF's %J variable
        log_pattern = (self.logging_config.filename_pattern.replace("{job_id}", "%J")
                      if self.logging_config is not None
                      else "%J_benchmark.log")
        # Replace {timestamp} with current timestamp if present
        if "{timestamp}" in log_pattern:
            log_pattern = log_pattern.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        cmd_parts = [
            'bsub',
            # Queue settings
            '-q', f'"{" ".join(hw.queue_names)}"',
            # GPU settings
            '-gpu', f'"num={hw.gpu.count}:j_exclusive=yes:gmem={hw.gpu.memory_gb}GB"' if hw.gpu.is_exclusive else f'"num={hw.gpu.count}:gmem={hw.gpu.memory_gb}GB"',
            # CPU and memory settings
            '-R', f'"rusage[mem={hw.memory_gb}GB]"',
            '-R', f'"affinity[core({hw.cpu_cores})]"',
            '-R', f'"span[hosts={hw.num_hosts}]"',
            '-n', str(hw.num_processes),
            '-M', f'{hw.memory_gb}GB',
            # Output log file
            '-o', f'"{output_dir}/{log_pattern}"'
        ]
        
        # Build the execution command
        exec_cmd = []
        
        # Add module loads
        for module in self.lsf_config.environment.modules:
            exec_cmd.extend(['module load', module, '&&']) 
        
        # Add the script and its arguments only if provided
        if script_path is not None:
            exec_cmd.extend(["python", str(script_path)])
            exec_cmd.extend(str(arg) for arg in args)
            # Join the execution command if we have any
            if exec_cmd:
                cmd_parts.append('"' + ' '.join(exec_cmd[:-1]) + '"')  # Remove trailing &&
            return cmd_parts
        else:
            # For dry-run, return a properly formatted shell command
            module_loads = ' '.join(exec_cmd[:-1]) if exec_cmd else ''  # Remove trailing &&
            return '\\\n    '.join([
                cmd_parts[0],  # bsub
                *[f'{cmd_parts[i]} {cmd_parts[i+1]}' for i in range(1, len(cmd_parts)-1, 2)],
                f'"{module_loads}"'
            ])

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