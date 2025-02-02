from abc import ABC, abstractmethod

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
