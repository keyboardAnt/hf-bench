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
