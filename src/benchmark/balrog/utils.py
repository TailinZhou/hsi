"""Balrog utility functions."""

import hashlib
import os
import time

import numpy as np


def get_unique_seed(process_num=None, episode_idx=0, base_seed=None, task_id=None, repeat_idx=0):
    """Generate a unique seed for an episode.

    When base_seed is not None (test mode), seeds are deterministic and
    reproducible (derived from base_seed + task_id + episode_idx + repeat_idx):
    `repeat_idx` rotates the map across test_repeats passes so repeat N>0 is a
    genuinely different map set, not a re-run of repeat 0's maps. When
    None (dev/val mode), uses PID + time for non-deterministic behavior
    (repeat_idx is ignored — dev already varies every episode via time_ns).
    """
    if base_seed is not None:
        unique_str = f"{base_seed}_{task_id}_{episode_idx}_{repeat_idx}"
    else:
        pid = os.getpid()
        time_ns = time.time_ns()
        unique_str = f"{pid}_{task_id}_{episode_idx}_{time_ns}"
    hashed = hashlib.sha256(unique_str.encode()).hexdigest()
    seed = int(hashed[:8], 16)
    return seed


def to_jsonable(x):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):  # numpy scalars (e.g., np.int64)
        return x.item()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x
