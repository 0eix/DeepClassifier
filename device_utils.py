import torch


def get_device(gpu=False):
    """
    Get the device of which to perform the training or the inference.

    Args:
        gpu (Bool): whether one should try to use the GPU or not.

    Returns:
        dataloaders (Device): the device on which to perform the training or
        the inference.
    """
    if not gpu:
        return torch.device("cpu")

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if not (has_cuda or has_mps):
        raise ValueError(
            "Neither CUDA nor MPS is available on this machine. Cannot use the GPU. "
            "Re-execute the program without the --gpu option or try on another machine."
        )

    return torch.device("mps" if has_mps else "cuda")

