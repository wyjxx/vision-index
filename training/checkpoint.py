"""Save and load optimizer state, random states, and epoch history."""

from pathlib import Path

import torch


def save_training_state(checkpoint_dir: Path, optimizer, epoch: int, run: dict):
    """Save the state required to continue after a completed validation epoch."""
    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "run": run,
    }
    # Publish the file only after writing it completely.
    temporary = checkpoint_dir / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(checkpoint_dir / "training_state.pt")


def load_training_state(checkpoint_dir: Path) -> dict:
    """Read a completed epoch's training state without loading tensors onto the GPU."""
    path = checkpoint_dir / "training_state.pt"
    if not path.is_file():
        raise ValueError("Checkpoint has no training_state.pt; its optimizer state cannot be restored")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not state["run"]["history"] or state["run"]["history"][-1]["epoch"] != state["epoch"]:
        raise ValueError("Checkpoint epoch and history do not match")
    return state
