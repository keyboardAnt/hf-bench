import os
import subprocess

from huggingface_hub import login

import wandb


def set_hf_cache_env():
    """
    Sets the environment variables for Hugging Face caching
    and creates the corresponding directories.
    """
    print("Cache location:", flush=True)
    hf_home = os.environ["HF_HOME"]  # Store in variable for clarity
    os.makedirs(hf_home, exist_ok=True)
    print(hf_home, flush=True)


def login_to_hf(token_env_var: str = "HF_ACCESS_TOKEN"):
    """
    Login to Hugging Face using an access token from the environment.
    """
    access_token = os.environ.get(token_env_var)
    if not access_token:
        raise ValueError(f"Environment variable {token_env_var} not found.")
    login(token=access_token)


def login_wandb():
    """
    Setup Weights & Biases for logging benchmark results.
    """
    print("Setting up W&B...", flush=True)
    wandb.login()
    print("W&B logged in", flush=True)


def log_hardware_info():
    """
    Logs hardware information including hostname, GPU, and CPU details to a file.
    """
    try:
        hostname = os.uname().nodename
        print(f"Hostname: {hostname}", flush=True)

        # Get GPU details using nvidia-smi
        gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(f"GPU Details:\n{gpu_info.stdout}", flush=True)

        # Get GPU memory usage
        gpu_memory_info = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        print(f"GPU Memory Usage:\n{gpu_memory_info.stdout}", flush=True)

        # Get CPU details using lscpu
        cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True)
        print(f"CPU Details:\n{cpu_info.stdout}", flush=True)

    except Exception as e:
        print(f"Error logging hardware information: {e}", flush=True)
