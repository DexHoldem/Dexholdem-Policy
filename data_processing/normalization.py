"""
Data normalization utilities.

All policy-relevant data (proprioception, actions) is normalized to [-1, 1]
using per-dimension min-max scaling.  Images are kept as uint8 and converted
to float in the encoder.

NormStats format
----------------
A nested dict of the form::

    {
        "action":   {"min": np.ndarray, "max": np.ndarray},
        "pos":      {"min": np.ndarray, "max": np.ndarray},
        "eef":      {"min": np.ndarray, "max": np.ndarray},
        ...
    }

The dict is JSON-serializable (after converting arrays to lists) and is
saved alongside every checkpoint.
"""

from __future__ import annotations

import numpy as np
from typing import Any


NormStats = dict[str, dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def get_data_stats(data: dict[str, np.ndarray]) -> NormStats:
    """
    Compute per-dimension min and max for every numeric array in *data*.

    Args:
        data: Dict mapping modality names to flat (N, dim) arrays.

    Returns:
        NormStats dict — a nested dict of {"min": ..., "max": ...} per key.
    """
    stats: NormStats = {}
    for key, arr in data.items():
        if not isinstance(arr, np.ndarray):
            continue
        if arr.dtype.kind not in ("f", "i", "u"):
            continue
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
        stats[key] = {
            "min": arr.min(axis=0).astype(np.float32),
            "max": arr.max(axis=0).astype(np.float32),
        }
    return stats


def merge_stats(stats_list: list[NormStats]) -> NormStats:
    """
    Merge normalization stats from multiple datasets (e.g., multi-task).

    Takes the element-wise min across all min arrays and the element-wise
    max across all max arrays.
    """
    merged: NormStats = {}
    for stats in stats_list:
        for key, stat in stats.items():
            if key not in merged:
                merged[key] = {
                    "min": stat["min"].copy(),
                    "max": stat["max"].copy(),
                }
            else:
                merged[key]["min"] = np.minimum(merged[key]["min"], stat["min"])
                merged[key]["max"] = np.maximum(merged[key]["max"], stat["max"])
    return merged


# ---------------------------------------------------------------------------
# Normalization / un-normalization
# ---------------------------------------------------------------------------

def normalize_data(
    data: np.ndarray,
    stat: dict[str, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Scale *data* to [-1, 1] using *stat*."""
    return (data - stat["min"]) / (stat["max"] - stat["min"] + eps) * 2 - 1


def unnormalize_data(
    ndata: np.ndarray,
    stat: dict[str, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Invert ``normalize_data``."""
    ndata = (ndata + 1) / 2
    return ndata * (stat["max"] - stat["min"] + eps) + stat["min"]


# ---------------------------------------------------------------------------
# Serialization helpers (for checkpoint saving)
# ---------------------------------------------------------------------------

def stats_to_json(stats: NormStats) -> dict[str, dict[str, list]]:
    """Convert numpy arrays to Python lists for JSON serialization."""
    return {
        key: {"min": stat["min"].tolist(), "max": stat["max"].tolist()}
        for key, stat in stats.items()
    }


def stats_from_json(json_stats: dict[str, dict[str, list]]) -> NormStats:
    """Reconstruct NormStats from JSON-loaded dict."""
    return {
        key: {
            "min": np.array(stat["min"], dtype=np.float32),
            "max": np.array(stat["max"], dtype=np.float32),
        }
        for key, stat in json_stats.items()
    }
