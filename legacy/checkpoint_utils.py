import os
import random
import tempfile

import numpy as np
import torch


def _to_cpu_byte_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.uint8, device="cpu")
    return torch.tensor(value, dtype=torch.uint8, device="cpu")


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_to_cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = state["cuda"]
        if isinstance(cuda_states, (list, tuple)):
            torch.cuda.set_rng_state_all([_to_cpu_byte_tensor(s) for s in cuda_states])
        else:
            torch.cuda.set_rng_state_all([_to_cpu_byte_tensor(cuda_states)])


def atomic_torch_save(payload, path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)