"""
RobotDataset — unified dataset for all policy models.

Every model receives the same batch dict from this dataset.  The format
matches the Batch Format defined in ``learning.base.PolicyModel``.

Sampling strategy
-----------------
Each dataset item is a *window* of consecutive timesteps extracted from a
flat memory-mapped buffer of all episodes.  Observations span ``obs_horizon``
timesteps; actions span ``pred_horizon`` timesteps starting at the same
position.  Windows that would cross episode boundaries are padded by
repeating the nearest valid timestep (edge padding).

Episode isolation
-----------------
When ``isolate_episodes=True``, the ``EpisodeAwareSampler`` ensures every
mini-batch contains only samples from the same episode, preventing the model
from seeing cross-episode transitions.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler


# ---------------------------------------------------------------------------
# Fast NPZ slice reader — bypasses np.load overhead for on-demand loading
# ---------------------------------------------------------------------------

# Per-worker cache of open ZipFile handles.  Re-opening a 400MB NPZ costs
# ~5ms (zip directory parsing); keeping the handle open makes subsequent
# reads from the same file nearly free.  Each DataLoader worker gets its
# own copy (threading.local) so there is no contention.
import threading
_zf_local = threading.local()


def _get_zipfile(path: Path) -> zipfile.ZipFile:
    """Return a cached ZipFile handle for *path* (per-thread/worker)."""
    cache = getattr(_zf_local, "cache", None)
    if cache is None:
        _zf_local.cache = {}
        cache = _zf_local.cache
    key = str(path)
    zf = cache.get(key)
    if zf is None:
        zf = zipfile.ZipFile(path, "r")
        cache[key] = zf
    return zf


def _read_npz_array_slice(
    path: Path, array_name: str, start: int, end: int,
) -> np.ndarray:
    """Read a slice [start:end] of a single array from an uncompressed NPZ.

    ~100x faster than ``np.load(path, mmap_mode='r')[name][start:end]``
    because it opens only the requested member and seeks directly to the
    needed bytes without parsing the entire zip directory for all arrays.
    Uses a per-worker ZipFile handle cache to avoid re-opening files.
    """
    member = f"{array_name}.npy"
    zf = _get_zipfile(path)
    with zf.open(member) as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        # Compute byte offset for the slice
        frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        # Skip to start frame
        if start > 0:
            f.read(start * frame_bytes)
        # Read requested frames
        n_frames = end - start
        raw = f.read(n_frames * frame_bytes)
        return np.frombuffer(raw, dtype=dtype).reshape(
            (n_frames, *shape[1:])
        ).copy()

from data_processing.loading import (
    iterate_dataset,
    iterate_dataset_lazy,
    LazyDatasetIndex,
    load_episode,
)
from data_processing.normalization import (
    NormStats,
    get_data_stats,
    normalize_data,
    merge_stats,
)


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Parameters shared between dataset construction and the training loop."""

    # Which modalities to include.
    representation_type: list[str] = field(default_factory=lambda: ["img", "pos"])

    # Which camera streams to include (0-indexed).
    camera_indices: list[int] = field(default_factory=lambda: [0, 1, 2])

    # Temporal windows.
    obs_horizon: int = 1
    pred_horizon: int = 64
    action_horizon: int = 32

    # If True, load raw images into RAM; otherwise keep on disk (slower).
    load_img: bool = True

    # Prevent training sequences from spanning episode boundaries.
    isolate_episodes: bool = True

    # Number of workers for parallel NPZ loading.
    n_load_workers: int = 4

    # Optional directory containing pre-extracted feature NPZ files that
    # mirror the data directory tree (output of precompute_features.py).
    # Feature file for data_dir/dataXXXX.npz is feature_dir/dataXXXX.npz.
    feature_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Sampling index helpers
# ---------------------------------------------------------------------------

def _create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int,
    pad_after: int,
    isolate: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Create (buffer_start, buffer_end, sample_start, sample_end) tuples for
    every valid sampling window across all episodes.

    Returns:
        indices:     (N, 4) int array.
        episode_ids: (N,) int array if isolate=True, else None.
    """
    indices, eids = [], []
    for ep_idx in range(len(episode_ends)):
        ep_start = int(episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
        ep_end   = int(episode_ends[ep_idx])
        ep_len   = ep_end - ep_start

        for start in range(-pad_before, ep_len - sequence_length + pad_after + 1):
            buf_start = max(start, 0) + ep_start
            buf_end   = min(start + sequence_length, ep_len) + ep_start
            samp_start = buf_start - (start + ep_start)
            samp_end   = sequence_length - ((start + sequence_length + ep_start) - buf_end)
            indices.append([buf_start, buf_end, samp_start, samp_end])
            eids.append(ep_idx)

    idx_arr = np.array(indices, dtype=np.int64)
    eid_arr = np.array(eids, dtype=np.int64) if isolate else None
    return idx_arr, eid_arr


def _sample_sequence(
    data: dict[str, Tensor | np.ndarray],
    buf_start: int,
    buf_end: int,
    samp_start: int,
    samp_end: int,
    seq_len: int,
) -> dict[str, Tensor]:
    """Extract a padded sequence from the flat buffer.

    Handles both shared-memory tensors and numpy memmap arrays transparently.
    For memmap arrays, only the requested slice is read from disk (OS paging).
    """
    out = {}
    for key, arr in data.items():
        if isinstance(arr, np.ndarray):
            # numpy / memmap — read only the needed slice, then convert.
            chunk = torch.from_numpy(np.array(arr[buf_start:buf_end]))
            seq = torch.zeros((seq_len, *arr.shape[1:]), dtype=chunk.dtype)
        else:
            chunk = arr[buf_start:buf_end]
            seq = torch.zeros((seq_len, *arr.shape[1:]), dtype=arr.dtype)
        seq[samp_start:samp_end] = chunk

        # Edge-pad left.
        if samp_start > 0:
            seq[:samp_start] = chunk[0]
        # Edge-pad right.
        if samp_end < seq_len:
            seq[samp_end:] = chunk[-1]

        out[key] = seq
    return out


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class RobotDataset(Dataset):
    """
    Unified dataset for robot imitation learning.

    Produces batch dicts compatible with ``PolicyModel.compute_loss``.

    Args:
        data_buffer:   Flat concatenated arrays (output of ``iterate_dataset``).
        episode_ends:  (num_episodes,) cumulative end indices.
        config:        Dataset configuration.
        norm_stats:    Pre-computed normalization stats.  If None, stats are
                       computed from *data_buffer*.
        instructions:  Per-episode instruction IDs.  Used when
                       ``use_instruction=True`` in the encoder config.
    """

    def __init__(
        self,
        data_buffer: dict[str, np.ndarray],
        episode_ends: np.ndarray,
        config: DatasetConfig,
        norm_stats: Optional[NormStats] = None,
        instructions: Optional[list[int | None]] = None,
    ):
        self.config = config
        self.episode_ends = episode_ends
        self.instructions = instructions or []

        # --- Build normalization stats ---
        # Exclude images, depth maps, and pre-computed feature arrays.
        # rgb_features_cam* are raw DinoV2 activations fed into a learned
        # projection — normalizing them is wrong and copying GBs is slow.
        stats_input = {
            k: v for k, v in data_buffer.items()
            if not k.startswith("rgb_cam")
            and not k.startswith("depth_cam")
            and not k.startswith("rgb_features_cam")
        }
        self.norm_stats: NormStats = norm_stats or get_data_stats(stats_input)

        # Compute global depth min/max (scalar, not per-pixel) so the
        # DepthEncoder can use data-driven normalization instead of
        # hardcoded constants.  Saved in the checkpoint alongside other stats.
        if "depth" not in self.norm_stats:
            depth_keys = [k for k in data_buffer if k.startswith("depth_cam")]
            if depth_keys:
                d_min = min(float(data_buffer[k].min()) for k in depth_keys)
                d_max = max(float(data_buffer[k].max()) for k in depth_keys)
                self.norm_stats["depth"] = {
                    "min": np.array([d_min], dtype=np.float32),
                    "max": np.array([d_max], dtype=np.float32),
                }

        # --- Normalize numeric arrays and store ---
        # Small arrays (proprioception, actions): normalize → shared-memory
        # tensors so DataLoader workers don't duplicate them via COW.
        # Large arrays (images, depth): keep as numpy memmap if already
        # mmap'd by iterate_dataset, or as regular numpy arrays.  These are
        # read-only and never normalized, so no copy is needed.
        # _sample_sequence handles both types transparently.
        self._normalized: dict[str, Tensor | np.ndarray] = {}
        for key, arr in data_buffer.items():
            is_mmap = isinstance(arr, np.memmap)
            if is_mmap:
                # Already memory-mapped — keep as-is.  The OS pages in only
                # the frames accessed per __getitem__ call.
                self._normalized[key] = arr
            elif key in self.norm_stats:
                arr = normalize_data(arr, self.norm_stats[key])
                t = torch.from_numpy(arr)
                t.share_memory_()
                self._normalized[key] = t
            else:
                t = torch.from_numpy(arr)
                t.share_memory_()
                self._normalized[key] = t

        # --- Sampling indices ---
        seq_len = config.obs_horizon + config.pred_horizon - 1
        self._indices, self._episode_ids = _create_indices(
            episode_ends,
            sequence_length=seq_len,
            pad_before=config.obs_horizon - 1,
            pad_after=config.pred_horizon - 1,
            isolate=config.isolate_episodes,
        )

        # For episodic isolation
        self.isolate_episodes = config.isolate_episodes
        self.episode_ids = self._episode_ids  # used by EpisodeAwareSampler

        # Offsets within the sampled sequence.
        self._obs_len = config.obs_horizon
        self._act_start = config.obs_horizon - 1
        self._act_end = config.obs_horizon - 1 + config.pred_horizon

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        buf_start, buf_end, samp_start, samp_end = self._indices[idx]
        seq_len = self.config.obs_horizon + self.config.pred_horizon - 1

        raw_seq = _sample_sequence(
            self._normalized,
            int(buf_start), int(buf_end),
            int(samp_start), int(samp_end),
            seq_len,
        )

        batch: dict[str, Tensor] = {}
        cam_idxs = self.config.camera_indices

        # ---- Observation window ----
        obs_window = {k: v[: self._obs_len] for k, v in raw_seq.items()}

        # Images — stack cameras into (obs_horizon, num_cams, H, W, 3).
        if any(f"rgb_cam{c}" in obs_window for c in cam_idxs):
            batch["rgb"] = torch.stack(
                [obs_window[f"rgb_cam{c}"] for c in cam_idxs if f"rgb_cam{c}" in obs_window],
                dim=1,   # (T, num_cams, H, W, 3)
            )

        if any(f"depth_cam{c}" in obs_window for c in cam_idxs):
            batch["depth"] = torch.stack(
                [obs_window[f"depth_cam{c}"] for c in cam_idxs if f"depth_cam{c}" in obs_window],
                dim=1,   # (T, num_cams, H, W)
            )

        # Pre-computed RGB features — stack cameras.
        feat_keys = [f"rgb_features_cam{c}" for c in cam_idxs]
        if any(k in obs_window for k in feat_keys):
            batch["rgb_features"] = torch.stack(
                [obs_window[k] for k in feat_keys if k in obs_window],
                dim=1,   # (T, num_cams, raw_dim)
            ).float()

        # Proprioceptive modalities.
        for key in ("pos", "eef", "hand_pos", "efforts", "velocity", "touch"):
            if key in obs_window:
                batch[key] = obs_window[key].float()

        # Per-step proprio over the action window for policies that consume
        # future proprioceptive context.
        # Keys are separate from the obs-window `pos`/`efforts` so other
        # models are unaffected.  Shape: (pred_horizon, dim).
        for key in ("pos", "efforts"):
            if key in raw_seq:
                batch[f"{key}_action"] = raw_seq[key][
                    self._act_start : self._act_end
                ].float()

        # ---- Action window ----
        act = raw_seq["action"][self._act_start : self._act_end]  # (pred_horizon, action_dim)
        batch["action"] = act.float()

        # ---- Instruction ----
        if self.instructions and self._episode_ids is not None:
            ep_id = int(self._episode_ids[idx])
            instr = self.instructions[ep_id]
            if instr is not None:
                batch["instruction"] = torch.tensor(instr, dtype=torch.long)

        return batch


# ---------------------------------------------------------------------------
# Lazy-loaded dataset (images loaded on-demand, near-zero RAM usage)
# ---------------------------------------------------------------------------

class LazyRobotDataset(Dataset):
    """Memory-efficient dataset that loads images on-demand from disk.

    Only proprioception / actions are kept in RAM (for normalization).
    Images, depth maps, and pre-computed features are read per-sample in
    ``__getitem__`` — each worker reads only the frames it needs.

    **Requires uncompressed NPZ files** for fast random access.  Run
    ``python workflow/npz_to_npy.py --data_dir <path>`` first.

    The public interface (``norm_stats``, ``isolate_episodes``,
    ``episode_ids``, ``__len__``, ``__getitem__`` output format) is
    identical to ``RobotDataset``, so they are interchangeable.
    """

    def __init__(
        self,
        lazy_index: LazyDatasetIndex,
        config: DatasetConfig,
        norm_stats: Optional[NormStats] = None,
    ):
        self.config = config
        self.episode_ends = lazy_index.episode_ends
        self.instructions = [e.instruction for e in lazy_index.episodes]
        self._lazy_index = lazy_index

        # --- Episode file lookup: global timestep → (episode_idx, local_t) ---
        self._ep_starts = np.zeros(len(lazy_index.episodes), dtype=np.int64)
        if len(lazy_index.episodes) > 1:
            self._ep_starts[1:] = lazy_index.episode_ends[:-1]

        # --- Normalization stats (from proprioception only — tiny) ---
        stats_input = {
            k: v for k, v in lazy_index.prop_buffer.items()
            if not k.startswith("rgb_cam")
            and not k.startswith("depth_cam")
            and not k.startswith("rgb_features_cam")
        }
        self.norm_stats: NormStats = norm_stats or get_data_stats(stats_input)

        # Compute global depth min/max by scanning episode files from disk.
        # Reads first + last frame of each episode to find range efficiently.
        if "depth" not in self.norm_stats and "depth" in lazy_index.representation_type:
            d_min, d_max = float("inf"), float("-inf")
            for ep in lazy_index.episodes:
                for ci in lazy_index.camera_indices:
                    try:
                        head = self._read_frames(
                            ep.path, f"depth_cam{ci}", 0, 1, is_dir=ep.is_dir)
                        tail = self._read_frames(
                            ep.path, f"depth_cam{ci}",
                            max(0, ep.length - 1), ep.length, is_dir=ep.is_dir)
                        d_min = min(d_min, float(head.min()), float(tail.min()))
                        d_max = max(d_max, float(head.max()), float(tail.max()))
                    except (KeyError, FileNotFoundError):
                        pass
            if d_min < float("inf"):
                self.norm_stats["depth"] = {
                    "min": np.array([d_min], dtype=np.float32),
                    "max": np.array([d_max], dtype=np.float32),
                }

        # --- Normalize proprioception and store as shared-memory tensors ---
        self._prop: dict[str, Tensor] = {}
        for key, arr in lazy_index.prop_buffer.items():
            if key in self.norm_stats:
                arr = normalize_data(arr, self.norm_stats[key])
            t = torch.from_numpy(arr)
            t.share_memory_()
            self._prop[key] = t

        # --- Sampling indices ---
        seq_len = config.obs_horizon + config.pred_horizon - 1
        self._indices, self._episode_ids = _create_indices(
            lazy_index.episode_ends,
            sequence_length=seq_len,
            pad_before=config.obs_horizon - 1,
            pad_after=config.pred_horizon - 1,
            isolate=config.isolate_episodes,
        )

        self.isolate_episodes = config.isolate_episodes
        self.episode_ids = self._episode_ids

        self._obs_len = config.obs_horizon
        self._act_start = config.obs_horizon - 1
        self._act_end = config.obs_horizon - 1 + config.pred_horizon

    def __len__(self) -> int:
        return len(self._indices)

    @staticmethod
    def _read_frames(path: Path, array_name: str,
                     start: int, end: int,
                     is_dir: bool | None = None) -> np.ndarray:
        """Read frames from either a .npy dir or .npz file.

        Retries once on transient I/O errors (stale mmap, file handle issues)
        to avoid crashing long-running training jobs.
        """
        if is_dir is None:
            is_dir = path.is_dir()
        for attempt in range(2):
            try:
                if is_dir:
                    npy_path = path / f"{array_name}.npy"
                    arr = np.load(str(npy_path), mmap_mode="r")
                    return np.array(arr[start:end])
                else:
                    return _read_npz_array_slice(path, array_name, start, end)
            except (OSError, IOError) as e:
                if attempt == 0:
                    # Retry once — clear cached handle if using NPZ
                    cache = getattr(_zf_local, "cache", {})
                    cache.pop(str(path), None)
                    continue
                raise RuntimeError(
                    f"Failed to read {array_name}[{start}:{end}] from {path} "
                    f"after 2 attempts: {e}"
                ) from e

    def _load_visual_frames(
        self, ep_idx: int, local_start: int, local_end: int,
    ) -> dict[str, torch.Tensor]:
        """Load visual data for frames [local_start, local_end) from one episode.

        Uses mmap for .npy directories (~0.1ms) or cached zipfile for .npz (~2ms).
        """
        ep = self._lazy_index.episodes[ep_idx]
        cam_idxs = self.config.camera_indices
        rep = self._lazy_index.representation_type
        out: dict[str, torch.Tensor] = {}

        # Load from the data (images / depth).
        # Only skip genuinely missing arrays (KeyError / FileNotFoundError).
        # Re-raise other errors (OSError, IOError) so they don't silently
        # produce inconsistent batches that crash the collator.
        need_img = "img" in rep
        need_depth = "depth" in rep
        data_is_dir = ep.is_dir
        if need_img or need_depth:
            if need_img:
                for cam in cam_idxs:
                    try:
                        frames = self._read_frames(
                            ep.path, f"images_cam{cam}",
                            local_start, local_end, is_dir=data_is_dir)
                        out[f"rgb_cam{cam}"] = torch.from_numpy(frames)
                    except (KeyError, FileNotFoundError):
                        pass
            if need_depth:
                for cam in cam_idxs:
                    try:
                        frames = self._read_frames(
                            ep.path, f"depth_cam{cam}",
                            local_start, local_end, is_dir=data_is_dir)
                        out[f"depth_cam{cam}"] = torch.from_numpy(
                            np.asarray(frames, dtype=np.float32)
                        )
                    except (KeyError, FileNotFoundError):
                        pass

        # Load pre-computed features
        if ep.feature_path is not None and ep.feature_path.exists():
            feat_is_dir = ep.feat_is_dir
            for cam in cam_idxs:
                feat_key = f"rgb_features_cam{cam}"
                try:
                    frames = self._read_frames(
                        ep.feature_path, feat_key,
                        local_start, local_end, is_dir=feat_is_dir)
                    out[feat_key] = torch.from_numpy(frames)
                except (KeyError, FileNotFoundError):
                    pass

        return out

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        buf_start, buf_end, samp_start, samp_end = self._indices[idx]
        buf_start, buf_end = int(buf_start), int(buf_end)
        samp_start, samp_end = int(samp_start), int(samp_end)
        seq_len = self.config.obs_horizon + self.config.pred_horizon - 1

        # --- Proprioception: from in-memory shared tensors ---
        prop_seq = _sample_sequence(
            self._prop, buf_start, buf_end, samp_start, samp_end, seq_len,
        )

        # --- Visual: lazy load ONLY the observation frames from disk ---
        # Images/depth/features are only used for the obs_horizon window,
        # not the full sequence. Loading only obs frames avoids reading
        # pred_horizon frames that would be immediately discarded.
        obs_len = self._obs_len
        ep_idx = int(np.searchsorted(self.episode_ends, buf_start, side="right"))
        ep_start_global = int(self._ep_starts[ep_idx])

        # Compute the valid-data intersection between the full padded sequence
        # and the observation window ``seq[:obs_len]``. This must match the
        # semantics of `_sample_sequence`, otherwise boundary samples can load
        # too many visual frames and fail during assignment.
        obs_samp_start = min(samp_start, obs_len)
        obs_samp_end = min(samp_end, obs_len)
        obs_real_len = max(obs_samp_end - obs_samp_start, 0)
        obs_local_start = buf_start - ep_start_global
        obs_local_end = obs_local_start + obs_real_len

        visual_frames = self._load_visual_frames(
            ep_idx, obs_local_start, obs_local_end)

        # Pad visual obs frames (edge-pad if window is at episode boundary).
        n_valid = obs_samp_end - obs_samp_start
        obs_window_vis: dict[str, Tensor] = {}
        for key, chunk in visual_frames.items():
            obs = torch.zeros((obs_len, *chunk.shape[1:]), dtype=chunk.dtype)
            obs[obs_samp_start:obs_samp_end] = chunk[:n_valid]
            if obs_samp_start > 0:
                obs[:obs_samp_start] = chunk[0]
            if obs_samp_end < obs_len:
                obs[obs_samp_end:] = chunk[-1]
            obs_window_vis[key] = obs

        # --- Assemble batch dict (same format as RobotDataset) ---
        batch: dict[str, Tensor] = {}
        cam_idxs = self.config.camera_indices

        obs_window_prop = {k: v[: self._obs_len] for k, v in prop_seq.items()}

        # Images
        if any(f"rgb_cam{c}" in obs_window_vis for c in cam_idxs):
            batch["rgb"] = torch.stack(
                [obs_window_vis[f"rgb_cam{c}"] for c in cam_idxs
                 if f"rgb_cam{c}" in obs_window_vis],
                dim=1,
            )

        if any(f"depth_cam{c}" in obs_window_vis for c in cam_idxs):
            batch["depth"] = torch.stack(
                [obs_window_vis[f"depth_cam{c}"] for c in cam_idxs
                 if f"depth_cam{c}" in obs_window_vis],
                dim=1,
            )

        # Pre-computed RGB features
        feat_keys = [f"rgb_features_cam{c}" for c in cam_idxs]
        if any(k in obs_window_vis for k in feat_keys):
            batch["rgb_features"] = torch.stack(
                [obs_window_vis[k] for k in feat_keys if k in obs_window_vis],
                dim=1,
            ).float()

        # Proprioceptive modalities
        for key in ("pos", "eef", "hand_pos", "efforts", "velocity", "touch"):
            if key in obs_window_prop:
                batch[key] = obs_window_prop[key].float()

        # Per-step proprio over the action window. Mirrors RobotDataset.
        for key in ("pos", "efforts"):
            if key in prop_seq:
                batch[f"{key}_action"] = prop_seq[key][
                    self._act_start : self._act_end
                ].float()

        # Action
        act = prop_seq["action"][self._act_start : self._act_end]
        batch["action"] = act.float()

        # Instruction
        if self.instructions and self._episode_ids is not None:
            ep_id = int(self._episode_ids[idx])
            instr = self.instructions[ep_id]
            if instr is not None:
                batch["instruction"] = torch.tensor(instr, dtype=torch.long)

        return batch


# ---------------------------------------------------------------------------
# Episode-aware sampler
# ---------------------------------------------------------------------------

class EpisodeAwareSampler(Sampler):
    """
    Yields batches where all samples belong to the same episode.

    This prevents the model from ever seeing a transition that spans two
    different demonstration episodes in a single forward pass.

    Requires ``dataset.isolate_episodes = True``.
    """

    def __init__(
        self,
        dataset: RobotDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if not dataset.isolate_episodes or dataset.episode_ids is None:
            raise ValueError(
                "EpisodeAwareSampler requires dataset.isolate_episodes=True."
            )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group sample indices by episode.
        ep_to_idx: dict[int, list[int]] = {}
        for i, ep in enumerate(dataset.episode_ids):
            ep_to_idx.setdefault(int(ep), []).append(i)
        self.ep_to_idx = ep_to_idx

    def __iter__(self) -> Iterator[list[int]]:
        episode_order = list(self.ep_to_idx.keys())
        if self.shuffle:
            np.random.shuffle(episode_order)

        for ep in episode_order:
            idxs = self.ep_to_idx[ep].copy()
            if self.shuffle:
                np.random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for idxs in self.ep_to_idx.values():
            n = len(idxs)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: str | Path,
    config: DatasetConfig,
    norm_stats: Optional[NormStats] = None,
) -> RobotDataset:
    """
    Load all episodes in *data_dir* and return a ``RobotDataset``.

    Args:
        data_dir:   Directory containing ``data*.npz`` files.
        config:     Dataset configuration.
        norm_stats: If provided, use these stats instead of computing them.
                    Pass the training set stats when constructing a validation
                    set so both are normalized consistently.
    """
    data_buffer, episode_ends, instructions = iterate_dataset(
        data_dir=data_dir,
        representation_type=config.representation_type,
        camera_indices=config.camera_indices,
        load_img=config.load_img,
        n_workers=config.n_load_workers,
        feature_dir=config.feature_dir,
    )
    return RobotDataset(
        data_buffer=data_buffer,
        episode_ends=episode_ends,
        config=config,
        norm_stats=norm_stats,
        instructions=instructions,
    )


def build_multitask_dataset(
    data_dirs: list[str | Path],
    config: DatasetConfig,
    norm_stats: Optional[NormStats] = None,
    feature_dirs: Optional[list[Optional[Path]]] = None,
) -> RobotDataset:
    """
    Load episodes from multiple task directories and concatenate them.

    Normalization stats are computed jointly across all tasks unless
    *norm_stats* is supplied.

    Args:
        feature_dirs: Per-task feature directories (same length as data_dirs).
                      Each entry is a leaf feature directory parallel to the
                      corresponding data_dir. Pass None for tasks without
                      precomputed features.
    """
    buffers, ends, instrs = [], [], []
    cumulative = 0
    per_task_stats = []

    for i, d in enumerate(data_dirs):
        feat_dir = feature_dirs[i] if feature_dirs else None
        buf, ep_ends, ep_instrs = iterate_dataset(
            data_dir=d,
            representation_type=config.representation_type,
            camera_indices=config.camera_indices,
            load_img=config.load_img,
            n_workers=config.n_load_workers,
            feature_dir=feat_dir,
        )
        buffers.append(buf)
        ends.append(ep_ends + cumulative)
        instrs.extend(ep_instrs)
        cumulative += ep_ends[-1]

        if norm_stats is None:
            numeric = {
                k: v for k, v in buf.items()
                if not k.startswith("rgb_cam")
                and not k.startswith("depth_cam")
                and not k.startswith("rgb_features_cam")
            }
            per_task_stats.append(get_data_stats(numeric))

    # Concatenate per-task buffers in RAM.
    all_keys = set().union(*[b.keys() for b in buffers])
    flat_buf: dict[str, np.ndarray] = {}

    for k in all_keys:
        parts = [b[k] for b in buffers if k in b]
        if not parts:
            continue
        flat_buf[k] = np.concatenate(parts, axis=0)

    all_ends = np.concatenate(ends)

    joint_stats = norm_stats or merge_stats(per_task_stats)

    return RobotDataset(
        data_buffer=flat_buf,
        episode_ends=all_ends,
        config=config,
        norm_stats=joint_stats,
        instructions=instrs,
    )


def build_dataset_lazy(
    data_dir: str | Path,
    config: DatasetConfig,
    norm_stats: Optional[NormStats] = None,
) -> LazyRobotDataset:
    """Lazy variant of ``build_dataset`` — images stay on disk."""
    index = iterate_dataset_lazy(
        data_dir=data_dir,
        representation_type=config.representation_type,
        camera_indices=config.camera_indices,
        n_workers=config.n_load_workers,
        feature_dir=config.feature_dir,
    )
    return LazyRobotDataset(index, config, norm_stats=norm_stats)


def build_multitask_dataset_lazy(
    data_dirs: list[str | Path],
    config: DatasetConfig,
    norm_stats: Optional[NormStats] = None,
    feature_dirs: Optional[list[Optional[Path]]] = None,
) -> LazyRobotDataset:
    """Lazy variant of ``build_multitask_dataset`` — images stay on disk."""
    all_episodes = []
    all_prop_parts: dict[str, list[np.ndarray]] = {}
    cumulative = 0
    all_ends = []
    per_task_stats = []

    for i, d in enumerate(data_dirs):
        feat_dir = feature_dirs[i] if feature_dirs else None
        index = iterate_dataset_lazy(
            data_dir=d,
            representation_type=config.representation_type,
            camera_indices=config.camera_indices,
            n_workers=config.n_load_workers,
            feature_dir=feat_dir,
        )
        all_episodes.extend(index.episodes)
        all_ends.append(index.episode_ends + cumulative)
        cumulative += index.episode_ends[-1]

        for k, v in index.prop_buffer.items():
            all_prop_parts.setdefault(k, []).append(v)

        if norm_stats is None:
            numeric = {
                k: v for k, v in index.prop_buffer.items()
                if not k.startswith("rgb_cam")
                and not k.startswith("depth_cam")
                and not k.startswith("rgb_features_cam")
            }
            per_task_stats.append(get_data_stats(numeric))

    merged_prop = {k: np.concatenate(parts, axis=0)
                   for k, parts in all_prop_parts.items()}
    merged_ends = np.concatenate(all_ends)
    joint_stats = norm_stats or merge_stats(per_task_stats)

    merged_index = LazyDatasetIndex(
        episodes=all_episodes,
        episode_ends=merged_ends,
        prop_buffer=merged_prop,
        representation_type=config.representation_type,
        camera_indices=config.camera_indices,
    )
    return LazyRobotDataset(merged_index, config, norm_stats=joint_stats)


def create_dataloader(
    dataset: RobotDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_episode_sampler: bool = False,
    sampler=None,
) -> DataLoader:
    """
    Wrap *dataset* in a DataLoader.

    Args:
        use_episode_sampler: If True, use ``EpisodeAwareSampler`` so each
                             mini-batch contains only one episode's samples.
                             Requires ``dataset.isolate_episodes=True``.
    """
    persistent = num_workers > 0

    if use_episode_sampler:
        sampler = EpisodeAwareSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    if sampler is not None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
