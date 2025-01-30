from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from hf_bench.schemas import HardwareResponse
import os

class BaseRunner(ABC):
    """Base class for all runners (local, LSF, SLURM, etc.).
    
    This abstract class defines the interface that all runners must implement.
    It provides methods for job submission, status checking, and hardware information collection.
    """
    
    def __init__(self, config):
        """Initialize the runner with configuration.
        
        Args:
            config: Configuration object containing execution parameters
        """
        self.config = config
    
    @abstractmethod
    def submit(self, script_path: str, *args, **kwargs) -> str:
        """Submit a job for execution.
        
        Args:
            script_path: Path to the Python script to execute
            *args: Additional arguments to pass to the script
            **kwargs: Additional keyword arguments for job submission
            
        Returns:
            str: A unique identifier for the submitted job
            
        Raises:
            RuntimeError: If job submission fails
        """
        pass
    
    @abstractmethod
    def get_status(self, job_id: str) -> str:
        """Get the current status of a submitted job.
        
        Args:
            job_id: The job identifier returned by submit()
            
        Returns:
            str: Current job status (e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED")
            
        Raises:
            KeyError: If job_id is not found
        """
        pass
    
    @abstractmethod
    def get_hardware_info(self, job_id: str) -> HardwareResponse:
        """Get hardware information for a submitted job.
        
        Args:
            job_id: The job identifier returned by submit()
            
        Returns:
            HardwareResponse: Object containing hardware information
            
        Raises:
            KeyError: If job_id is not found
        """
        pass
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a submitted job.
        
        Args:
            job_id: The job identifier returned by submit()
            
        Returns:
            bool: True if job was cancelled successfully, False otherwise
            
        Raises:
            KeyError: If job_id is not found
        """
        raise NotImplementedError("Cancel not implemented for this runner")
    
    def wait(self, job_id: str, timeout: float | None = None) -> bool:
        """Wait for a job to complete.
        
        Args:
            job_id: The job identifier returned by submit()
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            bool: True if job completed successfully, False if timeout or failure
            
        Raises:
            KeyError: If job_id is not found
            TimeoutError: If timeout is reached
        """
        raise NotImplementedError("Wait not implemented for this runner")
    
    def _validate_script_path(self, script_path: str | Path) -> Path:
        """Validate that the script path exists and is executable.
        
        Args:
            script_path: Path to the Python script
            
        Returns:
            Path: Validated Path object
            
        Raises:
            FileNotFoundError: If script doesn't exist
            PermissionError: If script isn't executable
        """
        path = Path(script_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Script is not executable: {path}")
        return path
