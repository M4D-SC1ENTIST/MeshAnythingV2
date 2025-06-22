import torch

def get_device():
    """
    Get the best available device for the current platform.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def is_mps_device(device):
    """Check if the device is MPS (Apple Silicon)"""
    return str(device).startswith('mps')

def is_cuda_device(device):
    """Check if the device is CUDA (NVIDIA GPU)"""
    return str(device).startswith('cuda')

def get_attention_implementation(device):
    """
    Get the appropriate attention implementation based on device.
    MPS doesn't support flash_attention_2, so use eager for MPS.
    """
    if is_mps_device(device):
        return "eager"
    else:
        return "flash_attention_2"

def should_use_bettertransformer(device):
    """
    Determine if bettertransformer should be used based on device.
    MPS may have compatibility issues with bettertransformer.
    """
    return not is_mps_device(device)

def get_dtype_for_device(device):
    """
    Get the appropriate dtype for the device.
    MPS has better compatibility with float32.
    """
    if is_mps_device(device):
        return torch.float32
    else:
        return None  # Use default dtype

def ensure_device_compatibility(tensor, device):
    """
    Ensure tensor is compatible with the target device.
    For MPS, convert to float32 if needed.
    """
    if is_mps_device(device) and tensor.dtype == torch.float16:
        return tensor.float()
    return tensor 