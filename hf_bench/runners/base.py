from abc import ABC, abstractmethod
from hf_bench.schemas import HardwareResponse

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
    def get_submit_command(self) -> str:
        """Generate the command string used to submit jobs.
        
        Returns:
            str: Command string that can be executed to submit a job
            
        This method should generate a command string that, when executed,
        would submit a job to the cluster/system. The command should include
        all necessary environment setup, resource requests, and execution parameters
        as specified in the runner's configuration.
        """
        pass
    
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
   