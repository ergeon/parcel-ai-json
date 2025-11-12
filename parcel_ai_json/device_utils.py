"""Device detection utilities for PyTorch models.

Provides automatic device selection for optimal performance across environments:
- CUDA (NVIDIA GPUs) for ECS GPU instances, EC2
- MPS (Apple Silicon) for local Mac development
- CPU as fallback for Docker, non-GPU instances
"""


def get_best_device() -> str:
    """Auto-detect best available device for PyTorch inference.

    Returns:
        "cuda" for NVIDIA GPUs (ECS g4dn instances, EC2 GPU instances)
        "mps" for Apple Silicon (local Mac development)
        "cpu" as fallback (Docker on Mac, non-GPU instances)
    """
    try:
        import torch

        # Check for NVIDIA CUDA (ECS GPU instances, EC2)
        if torch.cuda.is_available():
            return "cuda"

        # Check for Apple Silicon MPS (local Mac development)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    except ImportError:
        pass

    # Fallback to CPU (Docker on Mac, non-GPU instances)
    return "cpu"
