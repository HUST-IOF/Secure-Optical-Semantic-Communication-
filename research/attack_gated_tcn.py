#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep TCN partial known-prompt attack for 8808-D f -> z recovery.

Experiments for each number N of leaked FULL 8808-D f-z samples:
    1) baseline_attack
       Digital constellation transformation baseline. Select 16-QAM or 256-QAM.

    2) ours_same_state_attack
       Train the attacker using f-z pairs generated under physical state s1,
       and test on real prompts encoded under the same state s1.

    3) ours_diff_state_attack
       Reuse EXACTLY the same attacker trained under s1, but test on real
       prompts encoded under a different physical state s2.

Important:
    - The attacker input is always the COMPLETE 8808-D frequency sequence.
    - The attacker output is always the COMPLETE 8808-D prompt.
    - The attacker is a unified pointwise + deep dilated residual 1-D TCN.
    - No explicit 2-to-2 physical pairing is exposed to the attack network.
    - Training is controlled by EPOCHS, not by a fixed number of steps.
    - For N=0, there is no attack model. The encoded f itself is directly
      written as the prompt, representing "before attack".

Prompt format:
    U, V, U_scale, U_zero_point, V_scale, V_zero_point

For rank=8:
    U [77, 8] -> 616 bytes
    V [8, 1024] -> 8192 bytes
    total -> 8808 bytes

The output prompt folders are compatible with the existing generation.py
pipeline.
"""

import os
import re
import csv
import json
import glob
import math
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from numpy.lib.format import open_memmap

try:
    from f2VEC import process_full_vector
except ImportError:
    from ReflectionCreation.f2VEC import process_full_vector


PROMPT_KEYS = (
    "U", "V",
    "U_scale", "U_zero_point",
    "V_scale", "V_zero_point",
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Prompt I/O
# ============================================================

def prompt_dir(frame_path: str, rank: int, interval: int) -> Path:
    return Path(frame_path) / "results" / f"rank{rank}_interval{interval}"


def load_prompt_bytes(prompt_path: str):
    obj = torch.load(prompt_path, map_location="cpu", weights_only=False)

    missing = [k for k in PROMPT_KEYS if k not in obj]
    if missing:
        raise KeyError(f"{prompt_path} missing keys: {missing}")

    U = obj["U"].detach().cpu().to(torch.uint8).numpy()
    V = obj["V"].detach().cpu().to(torch.uint8).numpy()

    z = np.concatenate(
        [U.reshape(-1), V.reshape(-1)],
        axis=0,
    ).astype(np.uint8)

    meta = {
        "U_shape": tuple(U.shape),
        "V_shape": tuple(V.shape),
        "U_scale": obj["U_scale"],
        "U_zero_point": obj["U_zero_point"],
        "V_scale": obj["V_scale"],
        "V_zero_point": obj["V_zero_point"],
    }

    return z, meta


def _move_value(value, device: str):
    if torch.is_tensor(value):
        return value.detach().to(device)
    return value


def save_prompt_bytes(
    save_path: str,
    z_u8: np.ndarray,
    meta: Dict,
    save_device: str,
):
    z = np.asarray(z_u8, dtype=np.uint8).reshape(-1)

    u_size = int(np.prod(meta["U_shape"]))
    v_size = int(np.prod(meta["V_shape"]))
    expected = u_size + v_size

    if z.size != expected:
        raise ValueError(
            f"Recovered prompt length={z.size}, expected={expected}"
        )

    U = z[:u_size].reshape(meta["U_shape"])
    V = z[u_size:].reshape(meta["V_shape"])

    out = {
        "U": torch.from_numpy(U.copy()).to(save_device, dtype=torch.uint8),
        "V": torch.from_numpy(V.copy()).to(save_device, dtype=torch.uint8),
        "U_scale": _move_value(meta["U_scale"], save_device),
        "U_zero_point": _move_value(meta["U_zero_point"], save_device),
        "V_scale": _move_value(meta["V_scale"], save_device),
        "V_zero_point": _move_value(meta["V_zero_point"], save_device),
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_path)


def prepare_output_frame_folder(
    source_frame_path: str,
    destination_frame_path: str,
    rank: int,
    interval: int,
    overwrite: bool,
):
    """
    Copy the source frame folder while excluding the original results folder.
    Files required by generation.py are preserved.
    """
    src = Path(source_frame_path).resolve()
    dst = Path(destination_frame_path).resolve()

    if src == dst:
        raise ValueError("source and destination paths must differ")

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(
                f"{dst} exists. Use --overwrite to replace it."
            )

    def ignore_results(_dirpath, names):
        return {"results"} if "results" in names else set()

    shutil.copytree(src, dst, ignore=ignore_results)

    src_pd = prompt_dir(str(src), rank, interval)
    dst_pd = prompt_dir(str(dst), rank, interval)
    dst_pd.mkdir(parents=True, exist_ok=True)

    init_src = src_pd / "init.pth"
    if init_src.exists():
        shutil.copy2(init_src, dst_pd / "init.pth")

    return dst_pd


# ============================================================
# Physical MMF mapping
# ============================================================

def ensure_256x7(A):
    A = np.asarray(A, dtype=np.float64)

    if A.shape == (256, 7):
        return A

    if A.shape == (7, 256):
        return A.T

    raise ValueError(
        f"Expected physical response 256x7 or 7x256, got {A.shape}"
    )


def load_features_from_mat(mat_path: str):
    mat = sio.loadmat(mat_path)

    if "Data" not in mat:
        raise KeyError(f"'Data' missing in {mat_path}")

    features_256x7, _ = process_full_vector(mat["Data"])
    return ensure_256x7(features_256x7)


def build_pca2_params(features_256x7, eps=1e-12):
    A = ensure_256x7(features_256x7)

    mean_A = np.mean(A, axis=0, keepdims=True)
    centered = A - mean_A

    _, singular_values, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    V2 = Vt[:2, :].T
    B = centered @ V2

    b_min = np.min(B, axis=0, keepdims=True)
    b_max = np.max(B, axis=0, keepdims=True)

    B_norm = 2.0 * (B - b_min) / (
        b_max - b_min + eps
    ) - 1.0

    params = {
        "mean_A": mean_A,
        "V2": V2,
        "b_min": b_min,
        "b_max": b_max,
        "singular_values": singular_values,
        "eps": float(eps),
    }

    return B_norm, params


def all_2d_prompt_targets():
    values = np.arange(256, dtype=np.uint16)

    z0, z1 = np.meshgrid(
        values,
        values,
        indexing="ij",
    )

    z_pairs = np.stack(
        [z0.reshape(-1), z1.reshape(-1)],
        axis=1,
    ).astype(np.uint8)

    targets = (
        2.0 * z_pairs.astype(np.float64) / 255.0 - 1.0
    )

    return z_pairs, targets


def build_selected_table(features_256x7, gain: float):
    """
    For all 65536 possible 2-byte prompt targets, find the nearest physical
    frequency pair using the calibrated 2-D PCA response.
    """
    B_norm, params = build_pca2_params(features_256x7)
    K = B_norm.shape[0]

    i_idx, j_idx = np.meshgrid(
        np.arange(K),
        np.arange(K),
        indexing="ij",
    )

    i_idx = i_idx.reshape(-1)
    j_idx = j_idx.reshape(-1)

    all_frequency_pairs = np.stack(
        [i_idx, j_idx],
        axis=1,
    ).astype(np.uint16)

    pair_vectors = gain * (
        B_norm[i_idx] - B_norm[j_idx]
    )

    pair_vectors = np.clip(
        pair_vectors,
        -1.0,
        1.0,
    )

    z_pairs, targets = all_2d_prompt_targets()

    tree = cKDTree(pair_vectors)

    try:
        distance, nearest = tree.query(
            targets,
            k=1,
            workers=-1,
        )
    except TypeError:
        distance, nearest = tree.query(
            targets,
            k=1,
        )

    selected = all_frequency_pairs[nearest]
    matched = pair_vectors[nearest]

    z_hat = np.clip(
        np.round((matched + 1.0) * 127.5),
        0,
        255,
    ).astype(np.uint8)

    abs_error = np.abs(
        z_hat.astype(np.int16)
        - z_pairs.astype(np.int16)
    )

    params["gain"] = float(gain)

    stats = {
        "mean_lookup_loss": float(
            np.mean((distance ** 2) / 2.0)
        ),
        "max_lookup_loss": float(
            np.max((distance ** 2) / 2.0)
        ),
        "mean_lookup_mae8": float(
            np.mean(abs_error)
        ),
        "max_lookup_error8": int(
            np.max(abs_error)
        ),
    }

    return selected.astype(np.uint16), params, stats


def ensure_physical_table(
    mat_path: str,
    table_path: str,
    gain: float,
):
    table_path = Path(table_path)

    if table_path.exists():
        data = np.load(table_path, allow_pickle=True)
        return data["selected"].astype(np.int64)

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = load_features_from_mat(mat_path)

    selected, params, stats = build_selected_table(
        features,
        gain=gain,
    )

    np.savez_compressed(
        table_path,
        selected=selected.astype(np.uint16),
        mean_A=params["mean_A"],
        V2=params["V2"],
        b_min=params["b_min"],
        b_max=params["b_max"],
        singular_values=params["singular_values"],
        eps=np.asarray(params["eps"]),
        gain=np.asarray(params["gain"]),
    )

    print(f"[physical table] saved: {table_path}")
    print(json.dumps(stats, indent=2))

    return selected.astype(np.int64)


def collision_statistics(selected: np.ndarray, name: str):
    """
    Diagnostic only. This does not alter the attack.
    It reports whether multiple z pairs map to the same frequency pair.
    """
    selected = np.asarray(selected, dtype=np.int64)

    pair_id = (
        selected[:, 0] * 256
        + selected[:, 1]
    )

    _, counts = np.unique(
        pair_id,
        return_counts=True,
    )

    total = len(selected)
    unique_count = len(counts)

    print("\n" + "-" * 72)
    print(f"Physical lookup collision analysis: {name}")
    print("-" * 72)
    print(f"Total z pairs     : {total}")
    print(f"Unique f pairs    : {unique_count}")
    print(f"Unique ratio      : {unique_count / total:.6f}")
    print(f"Collision ratio   : {1.0 - unique_count / total:.6f}")
    print(f"Max z per same f  : {int(counts.max())}")
    print(f"Mean z per used f : {float(counts.mean()):.6f}")


def collect_data_files(data_root: str):
    pattern = os.path.join(
        data_root,
        "Data20260508_20km_*MHz_*deg_*.mat",
    )

    regex = re.compile(
        r"Data20260508_20km_(?P<freq>[\d.]+MHz)_"
        r"(?P<temp>-?\d+)deg_(?P<rep>\d+)\.mat$"
    )

    result = {}

    for path in sorted(glob.glob(pattern)):
        match = regex.match(
            os.path.basename(path)
        )

        if match is None:
            continue

        freq = match.group("freq")
        temp = int(match.group("temp"))
        rep = int(match.group("rep"))

        result.setdefault(freq, {}).setdefault(
            temp,
            {},
        )[rep] = path

    return result


def get_mat_path(
    data_root: str,
    freq: str,
    temp: int,
    rep: int,
):
    files = collect_data_files(data_root)

    try:
        return files[freq][temp][rep]
    except KeyError as exc:
        raise FileNotFoundError(
            f"No physical response for "
            f"freq={freq}, temp={temp}, rep={rep}"
        ) from exc


def physical_encode_numpy(
    z_u8: np.ndarray,
    selected: np.ndarray,
):
    """
    Encode 1-D [L] or 2-D [B,L] uint8 prompt vectors into uint8 frequency
    indices. The output length remains L.
    """
    z = np.asarray(z_u8, dtype=np.uint8)

    if z.ndim == 1:
        if z.size % 2 != 0:
            raise ValueError(
                "Physical encoder requires an even prompt length."
            )

        pairs = z.reshape(-1, 2).astype(np.int64)

        pair_indices = (
            pairs[:, 0] * 256
            + pairs[:, 1]
        )

        return selected[
            pair_indices
        ].reshape(-1).astype(np.uint8)

    if z.ndim == 2:
        if z.shape[1] % 2 != 0:
            raise ValueError(
                "Physical encoder requires an even prompt length."
            )

        pairs = z.reshape(
            z.shape[0],
            -1,
            2,
        ).astype(np.int64)

        pair_indices = (
            pairs[..., 0] * 256
            + pairs[..., 1]
        )

        return selected[
            pair_indices
        ].reshape(
            z.shape[0],
            -1,
        ).astype(np.uint8)

    raise ValueError(
        "z_u8 must be 1-D or 2-D."
    )


# ============================================================
# Baseline: selectable 16-QAM / 256-QAM digital transform
# ============================================================

def square_qam_constellation(M: int):
    if M not in (16, 256):
        raise ValueError(
            "Only 16-QAM and 256-QAM are supported."
        )

    side = int(round(math.sqrt(M)))

    levels = np.arange(
        -(side - 1),
        side,
        2,
        dtype=np.float64,
    )

    xx, yy = np.meshgrid(
        levels,
        levels,
        indexing="xy",
    )

    points = np.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        axis=1,
    )

    average_energy = np.mean(
        np.sum(points ** 2, axis=1)
    )

    points = points / math.sqrt(
        average_energy
    )

    return points


def build_constellation_permutation(
    M: int,
    rotation_deg: float,
    reflection: bool,
):
    """
    Adapted deterministic digital constellation transformation.

    A geometric QAM rotation/reflection is converted to a one-to-one symbol
    permutation using Hungarian assignment.

    This baseline is an adapted digital transform for a same-platform
    comparison; it is not claimed to reproduce a specific published system.
    """
    points = square_qam_constellation(M)

    theta = math.radians(
        rotation_deg
    )

    R = np.array(
        [
            [
                math.cos(theta),
                -math.sin(theta),
            ],
            [
                math.sin(theta),
                math.cos(theta),
            ],
        ],
        dtype=np.float64,
    )

    transformed = points @ R.T

    if reflection:
        transformed[:, 0] *= -1.0

    cost = np.sum(
        (
            transformed[:, None, :]
            - points[None, :, :]
        ) ** 2,
        axis=2,
    )

    row_ind, col_ind = linear_sum_assignment(
        cost
    )

    permutation = np.empty(
        M,
        dtype=np.int64,
    )

    permutation[row_ind] = col_ind

    inverse = np.empty_like(
        permutation
    )

    inverse[permutation] = np.arange(
        M,
        dtype=np.int64,
    )

    return permutation, inverse, points


def baseline_encode_numpy(
    z_u8: np.ndarray,
    qam_order: int,
    permutation: np.ndarray,
):
    """
    Preserve the 8808-D byte sequence length.

    256-QAM:
        one prompt byte -> one 256-QAM symbol index -> transformed byte.

    16-QAM:
        one prompt byte is split into high/low 4-bit nibbles.
        Both nibbles are transformed by the same 16-QAM permutation and packed
        back into one byte. Thus f and z remain the same length.
    """
    z = np.asarray(
        z_u8,
        dtype=np.uint8,
    )

    if qam_order == 256:
        return permutation[
            z.astype(np.int64)
        ].astype(np.uint8)

    if qam_order == 16:
        high = (
            z >> 4
        ).astype(np.int64)

        low = (
            z & 0x0F
        ).astype(np.int64)

        high_enc = permutation[
            high
        ].astype(np.uint8)

        low_enc = permutation[
            low
        ].astype(np.uint8)

        return (
            (high_enc << 4)
            | low_enc
        ).astype(np.uint8)

    raise ValueError(
        "qam_order must be 16 or 256"
    )


# ============================================================
# Deep dilated TCN attacker
# ============================================================

class DilatedResidualBlock(nn.Module):
    """
    Pre-norm residual block:
        GroupNorm -> GELU -> dilated Conv1d
        GroupNorm -> GELU -> Dropout -> dilated Conv1d
        + residual
    """

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int,
        dropout: float,
        num_groups: int,
    ):
        super().__init__()

        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_groups={num_groups}"
            )

        padding = dilation * (
            kernel_size - 1
        ) // 2

        self.norm1 = nn.GroupNorm(
            num_groups,
            channels,
        )

        self.act1 = nn.GELU()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.norm2 = nn.GroupNorm(
            num_groups,
            channels,
        )

        self.act2 = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        # Small residual branch initialization improves stability.
        nn.init.zeros_(
            self.conv2.bias
        )

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return residual + x


class GatedPointwiseTCNAttack(nn.Module):
    """
    Strong unified black-box attacker for complete 8808-D f -> complete 8808-D z.

    The SAME network is used for the baseline and proposed method.

    Architecture:
        z_point = A_point(f)
        delta_z = A_TCN(f)
        alpha   = sigmoid(context_logit)
        z_hat   = z_point + alpha * delta_z

    Pointwise branch:
        learns a symbol-wise mapping f_i -> z_i. This is well matched to
        deterministic digital baselines such as the packed 16-QAM / 256-QAM
        transform.

    Contextual branch:
        a deep dilated TCN learns only the residual information that cannot be
        explained by the pointwise mapping. The network is NOT told the
        physical 2-to-2 pairing rule.

    The contextual gate is initialized small so that a simple pointwise mapping
    is not immediately polluted by the TCN. If contextual information is truly
    useful, training can increase the gate automatically.

    No secret key, physical response matrix, lookup table, calibrated PCA
    parameters, or explicit physical pairing information is provided.
    """

    def __init__(
        self,
        sequence_length: int = 8808,
        channels: int = 128,
        num_blocks: int = 12,
        kernel_size: int = 3,
        dropout: float = 0.05,
        num_groups: int = 8,
        max_dilation_power: int = 11,
        initial_context_logit: float = -3.0,
    ):
        super().__init__()

        self.sequence_length = int(sequence_length)
        self.channels = int(channels)

        # ----------------------------------------------------
        # Main pointwise branch.
        # 256 possible observed byte/frequency values -> normalized z value.
        # It starts from zero and learns only from leaked f-z observations.
        # ----------------------------------------------------
        self.pointwise_lookup = nn.Embedding(
            num_embeddings=256,
            embedding_dim=1,
        )
        nn.init.zeros_(self.pointwise_lookup.weight)

        # ----------------------------------------------------
        # Residual contextual TCN branch.
        # ----------------------------------------------------
        self.embedding = nn.Embedding(
            256,
            channels,
        )

        self.input_projection = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
        )

        dilation_cycle = [
            2 ** p
            for p in range(max_dilation_power + 1)
        ]

        self.dilations = [
            dilation_cycle[i % len(dilation_cycle)]
            for i in range(num_blocks)
        ]

        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    channels=channels,
                    dilation=d,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    num_groups=num_groups,
                )
                for d in self.dilations
            ]
        )

        self.output_norm = nn.GroupNorm(
            num_groups,
            channels,
        )
        self.output_act = nn.GELU()

        self.output_projection = nn.Conv1d(
            channels,
            1,
            kernel_size=1,
        )

        # Residual branch begins at zero.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        # alpha = sigmoid(-3) ~= 0.047 initially.
        self.context_logit = nn.Parameter(
            torch.tensor(
                float(initial_context_logit),
                dtype=torch.float32,
            )
        )

    def forward(self, f_indices, return_components: bool = False):
        if f_indices.ndim != 2:
            raise ValueError(
                f"Expected [B,L], got {tuple(f_indices.shape)}"
            )

        if f_indices.shape[1] != self.sequence_length:
            raise ValueError(
                f"Input length={f_indices.shape[1]}, "
                f"expected={self.sequence_length}"
            )

        f_indices = f_indices.long()

        # ----------------------------------------------------
        # Main pointwise prediction.
        # ----------------------------------------------------
        z_point = self.pointwise_lookup(
            f_indices
        ).squeeze(-1)

        # ----------------------------------------------------
        # Contextual residual prediction.
        # ----------------------------------------------------
        x = self.embedding(
            f_indices
        ).transpose(1, 2)

        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_norm(x)
        x = self.output_act(x)

        delta_z = self.output_projection(
            x
        ).squeeze(1)

        alpha = torch.sigmoid(
            self.context_logit
        )

        z_hat = (
            z_point
            + alpha * delta_z
        )

        if return_components:
            return (
                z_hat,
                z_point,
                delta_z,
                alpha,
            )

        return z_hat


def target_bytes_to_norm(
    z_u8: torch.Tensor,
):
    return (
        z_u8.float() / 127.5 - 1.0
    )


def norm_to_prompt_bytes(
    z_norm: torch.Tensor,
):
    # Clamp only during inference / metric conversion.
    z_norm = torch.clamp(
        z_norm,
        -1.0,
        1.0,
    )

    return torch.clamp(
        torch.round(
            (z_norm + 1.0) * 127.5
        ),
        0,
        255,
    ).to(torch.uint8)


@torch.no_grad()
def predict_prompt_bytes(
    model: nn.Module,
    f_u8: np.ndarray,
    device: str,
):
    model.eval()

    f = torch.from_numpy(
        np.asarray(
            f_u8,
            dtype=np.uint8,
        ).reshape(
            1,
            -1,
        ).astype(np.int64)
    ).to(
        device=device,
        dtype=torch.long,
    )

    pred_norm = model(f)

    return norm_to_prompt_bytes(
        pred_norm
    )[0].cpu().numpy()


# ============================================================
# Large nested random f-z training datasets
# ============================================================

def ensure_master_random_z(
    path: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
    chunk_size: int,
):
    """
    Store the master random prompt dataset as uint8 .npy memmap so that
    N=100000 does not require loading the complete array into RAM.
    """
    path = Path(path)

    if path.exists():
        arr = np.load(
            path,
            mmap_mode="r",
        )

        expected = (
            num_samples,
            sequence_length,
        )

        if arr.shape != expected:
            raise ValueError(
                f"Existing {path} has shape={arr.shape}, "
                f"expected={expected}"
            )

        return arr

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mmap = open_memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            num_samples,
            sequence_length,
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    for start in range(
        0,
        num_samples,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            num_samples,
        )

        mmap[start:end] = rng.integers(
            0,
            256,
            size=(
                end - start,
                sequence_length,
            ),
            dtype=np.uint8,
        )

        print(
            f"[master z] {end}/{num_samples}"
        )

    mmap.flush()

    return np.load(
        path,
        mmap_mode="r",
    )


def ensure_encoded_master(
    output_path: str,
    z_master,
    encoder_fn,
    num_samples: int,
    sequence_length: int,
    chunk_size: int,
):
    """
    Build a uint8 .npy memmap for encoded f without holding the full dataset
    in RAM.
    """
    output_path = Path(
        output_path
    )

    if output_path.exists():
        arr = np.load(
            output_path,
            mmap_mode="r",
        )

        expected = (
            num_samples,
            sequence_length,
        )

        if arr.shape != expected:
            raise ValueError(
                f"Existing {output_path} has shape={arr.shape}, "
                f"expected={expected}"
            )

        return arr

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mmap = open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            num_samples,
            sequence_length,
        ),
    )

    for start in range(
        0,
        num_samples,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            num_samples,
        )

        z_chunk = np.asarray(
            z_master[start:end],
            dtype=np.uint8,
        )

        mmap[start:end] = encoder_fn(
            z_chunk
        )

        print(
            f"[encoded master] "
            f"{output_path.name}: "
            f"{end}/{num_samples}"
        )

    mmap.flush()

    return np.load(
        output_path,
        mmap_mode="r",
    )


# ============================================================
# Epoch-based attack training
# ============================================================

def train_attack_by_epochs(
    experiment_name: str,
    f_train_u8,
    z_train_u8,
    n_leak: int,
    model_path: str,
    sequence_length: int,
    device: str,
    epochs: int,
    min_optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    channels: int,
    num_blocks: int,
    kernel_size: int,
    dropout: float,
    num_groups: int,
    max_dilation_power: int,
    pointwise_aux_weight: float,
    context_reg_weight: float,
    initial_context_logit: float,
    seed: int,
    log_batch_every: int,
    force_retrain: bool,
):
    """
    Train from scratch on the first n_leak FULL 8808-D f-z samples.

    Main objective:
        L_total = MSE(z_hat, z)

    Auxiliary objective:
        L_point = MSE(z_point, z)

    Small residual regularization:
        L_ctx = mean(delta_z^2)

    Optimization objective:
        L = L_total
            + pointwise_aux_weight * L_point
            + context_reg_weight * L_ctx

    This explicitly forces the symbol-wise branch to learn any simple
    f_i -> z_i relation, while allowing the TCN to learn only residual
    contextual structure.

    Training budget:
        actual_epochs = max(
            epochs,
            ceil(min_optimizer_steps / steps_per_epoch)
        )
    """
    if n_leak <= 0:
        return None

    if n_leak > len(f_train_u8):
        raise ValueError(
            f"n_leak={n_leak} exceeds master dataset "
            f"size={len(f_train_u8)}"
        )

    set_seed(seed)

    model = GatedPointwiseTCNAttack(
        sequence_length=sequence_length,
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        num_groups=num_groups,
        max_dilation_power=max_dilation_power,
        initial_context_logit=initial_context_logit,
    ).to(device)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and not force_retrain:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        print(
            f"[load] {experiment_name}, "
            f"N={n_leak}: {model_path}"
        )
        return model

    steps_per_epoch = int(
        math.ceil(n_leak / batch_size)
    )

    min_epochs_from_steps = int(
        math.ceil(
            max(min_optimizer_steps, 0)
            / max(steps_per_epoch, 1)
        )
    )

    actual_epochs = max(
        int(epochs),
        min_epochs_from_steps,
    )

    total_optimizer_steps = (
        actual_epochs * steps_per_epoch
    )

    # Do not regularize the small lookup table or the scalar context gate.
    pointwise_params = list(
        model.pointwise_lookup.parameters()
    )
    gate_params = [
        model.context_logit
    ]

    excluded_ids = {
        id(p)
        for p in pointwise_params + gate_params
    }

    context_params = [
        p
        for p in model.parameters()
        if id(p) not in excluded_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": pointwise_params,
                "weight_decay": 0.0,
            },
            {
                "params": gate_params,
                "weight_decay": 0.0,
            },
            {
                "params": context_params,
                "weight_decay": weight_decay,
            },
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_optimizer_steps, 1),
        eta_min=max(
            learning_rate * 0.05,
            1e-6,
        ),
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    rng = np.random.default_rng(seed)
    global_step = 0

    print("\n" + "=" * 96)
    print(
        f"Training {experiment_name} | "
        f"N={n_leak} | requested_epochs={epochs} | "
        f"actual_epochs={actual_epochs} | "
        f"steps/epoch={steps_per_epoch} | "
        f"total_steps={total_optimizer_steps} | "
        f"batch={batch_size}"
    )
    print(
        f"Gated Pointwise + Residual TCN | "
        f"channels={channels}, blocks={num_blocks}, "
        f"dilations={model.dilations}"
    )
    print(
        f"point_aux={pointwise_aux_weight:g} | "
        f"context_reg={context_reg_weight:g} | "
        f"initial_alpha="
        f"{torch.sigmoid(model.context_logit).item():.5f}"
    )
    print("=" * 96)

    for epoch in range(1, actual_epochs + 1):
        model.train()

        permutation = rng.permutation(n_leak)

        epoch_objective_sum = 0.0
        epoch_total_mse_sum = 0.0
        epoch_point_mse_sum = 0.0

        epoch_total_abs_error_sum = 0.0
        epoch_point_abs_error_sum = 0.0
        epoch_value_count = 0

        for batch_number, start in enumerate(
            range(0, n_leak, batch_size),
            start=1,
        ):
            batch_indices = permutation[
                start:start + batch_size
            ]

            f_np = np.asarray(
                f_train_u8[batch_indices],
                dtype=np.uint8,
            )

            z_np = np.asarray(
                z_train_u8[batch_indices],
                dtype=np.uint8,
            )

            f_batch = torch.from_numpy(
                f_np.astype(np.int64, copy=False)
            ).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            z_batch_u8 = torch.from_numpy(
                z_np.astype(np.int64, copy=False)
            ).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            z_target = target_bytes_to_norm(
                z_batch_u8
            )

            (
                pred,
                pred_point,
                delta_context,
                alpha,
            ) = model(
                f_batch,
                return_components=True,
            )

            total_mse = criterion(
                pred,
                z_target,
            )

            point_mse = criterion(
                pred_point,
                z_target,
            )

            context_reg = torch.mean(
                delta_context ** 2
            )

            loss = (
                total_mse
                + pointwise_aux_weight
                * point_mse
                + context_reg_weight
                * context_reg
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            global_step += 1

            batch_size_actual = len(
                batch_indices
            )

            epoch_objective_sum += (
                loss.item()
                * batch_size_actual
            )

            epoch_total_mse_sum += (
                total_mse.item()
                * batch_size_actual
            )

            epoch_point_mse_sum += (
                point_mse.item()
                * batch_size_actual
            )

            with torch.no_grad():
                pred_u8 = norm_to_prompt_bytes(
                    pred
                )

                point_u8 = norm_to_prompt_bytes(
                    pred_point
                )

                total_abs_error = torch.sum(
                    torch.abs(
                        pred_u8.float()
                        - z_batch_u8.float()
                    )
                ).item()

                point_abs_error = torch.sum(
                    torch.abs(
                        point_u8.float()
                        - z_batch_u8.float()
                    )
                ).item()

                byte_acc = (
                    pred_u8.long()
                    == z_batch_u8.long()
                ).float().mean().item()

                point_byte_acc = (
                    point_u8.long()
                    == z_batch_u8.long()
                ).float().mean().item()

                hi_acc = None
                lo_acc = None

                if experiment_name.startswith(
                    "baseline_qam16"
                ):
                    pred_long = pred_u8.long()
                    target_long = z_batch_u8.long()

                    pred_hi = torch.bitwise_right_shift(
                        pred_long,
                        4,
                    )
                    pred_lo = torch.bitwise_and(
                        pred_long,
                        15,
                    )

                    target_hi = torch.bitwise_right_shift(
                        target_long,
                        4,
                    )
                    target_lo = torch.bitwise_and(
                        target_long,
                        15,
                    )

                    hi_acc = (
                        pred_hi == target_hi
                    ).float().mean().item()

                    lo_acc = (
                        pred_lo == target_lo
                    ).float().mean().item()

            epoch_total_abs_error_sum += (
                total_abs_error
            )

            epoch_point_abs_error_sum += (
                point_abs_error
            )

            epoch_value_count += (
                batch_size_actual
                * sequence_length
            )

            if (
                log_batch_every > 0
                and global_step % log_batch_every == 0
            ):
                total_mae8 = (
                    total_abs_error
                    / (
                        batch_size_actual
                        * sequence_length
                    )
                )

                point_mae8 = (
                    point_abs_error
                    / (
                        batch_size_actual
                        * sequence_length
                    )
                )

                message = (
                    f"{experiment_name:20s} | "
                    f"N={n_leak:7d} | "
                    f"epoch={epoch:04d}/{actual_epochs:04d} | "
                    f"step={global_step:06d}/"
                    f"{total_optimizer_steps:06d} | "
                    f"obj={loss.item():.6f} | "
                    f"total_MAE8={total_mae8:.3f} | "
                    f"point_MAE8={point_mae8:.3f} | "
                    f"alpha={alpha.item():.5f} | "
                    f"byte_acc={byte_acc:.4f} | "
                    f"point_byte_acc={point_byte_acc:.4f}"
                )

                if hi_acc is not None:
                    message += (
                        f" | hi4_acc={hi_acc:.4f}"
                        f" | lo4_acc={lo_acc:.4f}"
                    )

                print(message)

        mean_objective = (
            epoch_objective_sum / n_leak
        )

        mean_total_mse = (
            epoch_total_mse_sum / n_leak
        )

        mean_point_mse = (
            epoch_point_mse_sum / n_leak
        )

        mean_total_mae8 = (
            epoch_total_abs_error_sum
            / max(epoch_value_count, 1)
        )

        mean_point_mae8 = (
            epoch_point_abs_error_sum
            / max(epoch_value_count, 1)
        )

        current_lr = optimizer.param_groups[0]["lr"]
        current_alpha = torch.sigmoid(
            model.context_logit
        ).item()

        print_epoch = (
            actual_epochs <= 50
            or epoch == 1
            or epoch == actual_epochs
            or epoch % max(actual_epochs // 20, 1) == 0
        )

        if print_epoch:
            print(
                f"{experiment_name:20s} | "
                f"N={n_leak:7d} | "
                f"EPOCH {epoch:04d}/"
                f"{actual_epochs:04d} | "
                f"obj={mean_objective:.6f} | "
                f"total_mse={mean_total_mse:.6f} | "
                f"point_mse={mean_point_mse:.6f} | "
                f"total_MAE8={mean_total_mae8:.3f} | "
                f"point_MAE8={mean_point_mae8:.3f} | "
                f"alpha={current_alpha:.5f} | "
                f"lr={current_lr:.3e}"
            )

        # Select best checkpoint by the actual full-output MSE.
        if mean_total_mse < best_loss:
            best_loss = mean_total_mse

            best_state = {
                key: value.detach().cpu()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No model state was saved during training."
        )

    model.load_state_dict(
        best_state
    )

    final_alpha = torch.sigmoid(
        model.context_logit.detach().cpu()
    ).item()

    torch.save(
        {
            "model": best_state,
            "experiment_name": experiment_name,
            "n_leak": n_leak,
            "best_train_total_mse": best_loss,
            "sequence_length": sequence_length,
            "config": {
                "architecture": "GatedPointwiseTCNAttack",
                "prediction": (
                    "z_point + sigmoid(context_logit) * delta_tcn"
                ),
                "pointwise_branch": (
                    "learnable 256-entry symbol-wise black-box mapping"
                ),
                "context_branch": (
                    "deep dilated residual TCN"
                ),
                "pointwise_aux_weight": (
                    pointwise_aux_weight
                ),
                "context_reg_weight": (
                    context_reg_weight
                ),
                "initial_context_logit": (
                    initial_context_logit
                ),
                "final_context_alpha": (
                    final_alpha
                ),
                "channels": channels,
                "num_blocks": num_blocks,
                "kernel_size": kernel_size,
                "dropout": dropout,
                "num_groups": num_groups,
                "max_dilation_power": (
                    max_dilation_power
                ),
                "dilations": model.dilations,
                "requested_epochs": epochs,
                "actual_epochs": actual_epochs,
                "min_optimizer_steps": (
                    min_optimizer_steps
                ),
                "total_optimizer_steps": (
                    total_optimizer_steps
                ),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss": (
                    "MSE(total,z) + lambda_point*MSE(point,z) "
                    "+ lambda_ctx*mean(delta_tcn^2)"
                ),
            },
        },
        model_path,
    )

    print(
        f"[save] {experiment_name}, "
        f"N={n_leak}: {model_path} | "
        f"final_alpha={final_alpha:.5f}"
    )

    return model


# ============================================================
# Evaluation and prompt output
# ============================================================

def mae8(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    reference = reference.astype(
        np.int16
    )

    estimate = estimate.astype(
        np.int16
    )

    return float(
        np.mean(
            np.abs(
                reference - estimate
            )
        )
    )


def mse_norm(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    reference = (
        reference.astype(np.float32)
        / 127.5 - 1.0
    )

    estimate = (
        estimate.astype(np.float32)
        / 127.5 - 1.0
    )

    return float(
        np.mean(
            (
                reference
                - estimate
            ) ** 2
        )
    )


def create_output_folders(
    source_frame_path: str,
    output_root: str,
    n_leak: int,
    rank: int,
    interval: int,
    overwrite: bool,
):
    base = Path(
        output_root
    ) / f"N{n_leak:06d}"

    condition_names = (
        "baseline_attack",
        "ours_same_state_attack",
        "ours_diff_state_attack",
    )

    result = {}

    for name in condition_names:
        frame_folder = base / name

        prompt_folder = (
            prepare_output_frame_folder(
                source_frame_path=source_frame_path,
                destination_frame_path=str(
                    frame_folder
                ),
                rank=rank,
                interval=interval,
                overwrite=overwrite,
            )
        )

        result[name] = {
            "frame_folder": frame_folder,
            "prompt_folder": prompt_folder,
        }

    return result


def evaluate_real_prompts(
    source_frame_path: str,
    output_folders: Dict,
    rank: int,
    interval: int,
    save_device: str,
    sequence_length: int,
    n_leak: int,
    qam_order: int,
    baseline_permutation: np.ndarray,
    same_selected: np.ndarray,
    diff_selected: np.ndarray,
    baseline_model,
    ours_model,
    attack_device: str,
):
    """
    N=0:
        No inversion model is used.
        The encoded f is directly stored as a .prompt file.

    N>0:
        baseline_model: baseline f -> z
        ours_model:     s1 physical f -> z

        The same ours_model is used for both same-state and different-state
        testing. The difference-state branch changes only the physical encoder.
    """
    source_prompt_dir = prompt_dir(
        source_frame_path,
        rank,
        interval,
    )

    prompt_files = sorted(
        source_prompt_dir.glob(
            "frame_*.prompt"
        )
    )

    if not prompt_files:
        raise FileNotFoundError(
            f"No frame_*.prompt in {source_prompt_dir}"
        )

    rows = []

    for index, prompt_path in enumerate(
        prompt_files
    ):
        z_true, meta = load_prompt_bytes(
            str(prompt_path)
        )

        if z_true.size != sequence_length:
            raise ValueError(
                f"{prompt_path}: "
                f"prompt length={z_true.size}, "
                f"expected={sequence_length}"
            )

        # ----------------------------------------------------
        # Baseline encoded frequency sequence.
        # ----------------------------------------------------
        f_baseline = baseline_encode_numpy(
            z_true,
            qam_order=qam_order,
            permutation=baseline_permutation,
        )

        # ----------------------------------------------------
        # Proposed encoded frequency sequences.
        # ----------------------------------------------------
        f_same = physical_encode_numpy(
            z_true,
            same_selected,
        )

        f_diff = physical_encode_numpy(
            z_true,
            diff_selected,
        )

        # ----------------------------------------------------
        # Before attack: write f directly as prompt.
        # After attack: write neural f->z output as prompt.
        # ----------------------------------------------------
        if n_leak == 0:
            z_baseline = f_baseline.copy()
            z_same = f_same.copy()
            z_diff = f_diff.copy()

        else:
            z_baseline = predict_prompt_bytes(
                baseline_model,
                f_baseline,
                attack_device,
            )

            z_same = predict_prompt_bytes(
                ours_model,
                f_same,
                attack_device,
            )

            z_diff = predict_prompt_bytes(
                ours_model,
                f_diff,
                attack_device,
            )

        outputs = {
            "baseline_attack": z_baseline,
            "ours_same_state_attack": z_same,
            "ours_diff_state_attack": z_diff,
        }

        for condition, z_hat in outputs.items():
            save_path = (
                output_folders[
                    condition
                ]["prompt_folder"]
                / prompt_path.name
            )

            save_prompt_bytes(
                save_path=str(
                    save_path
                ),
                z_u8=z_hat,
                meta=meta,
                save_device=save_device,
            )

            rows.append(
                {
                    "n_leak": n_leak,
                    "prompt_file": prompt_path.name,
                    "condition": condition,
                    "prompt_mae8": mae8(
                        z_true,
                        z_hat,
                    ),
                    "prompt_mse_norm": mse_norm(
                        z_true,
                        z_hat,
                    ),
                }
            )

        print(
            f"[real {index + 1:03d}/"
            f"{len(prompt_files):03d}] "
            f"N={n_leak:7d} | "
            f"{prompt_path.name} | "
            f"baseline_MAE8="
            f"{mae8(z_true, z_baseline):.2f} | "
            f"same_MAE8="
            f"{mae8(z_true, z_same):.2f} | "
            f"diff_MAE8="
            f"{mae8(z_true, z_diff):.2f}"
        )

    return rows


def aggregate_rows(rows):
    grouped = {}

    for row in rows:
        key = (
            row["n_leak"],
            row["condition"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    result = []

    for (
        n_leak,
        condition,
    ), items in sorted(
        grouped.items()
    ):
        result.append(
            {
                "n_leak": int(
                    n_leak
                ),
                "condition": condition,
                "num_prompts": len(
                    items
                ),
                "mean_prompt_mae8": float(
                    np.mean(
                        [
                            item[
                                "prompt_mae8"
                            ]
                            for item in items
                        ]
                    )
                ),
                "mean_prompt_mse_norm": float(
                    np.mean(
                        [
                            item[
                                "prompt_mse_norm"
                            ]
                            for item in items
                        ]
                    )
                ),
            }
        )

    return result


def write_csv(
    path: str,
    rows,
    fieldnames,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text: str):
    values = sorted(
        set(
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        )
    )

    if not values:
        raise ValueError(
            "--leak_counts cannot be empty"
        )

    if values[0] < 0:
        raise ValueError(
            "Leak counts must be >= 0"
        )

    return values


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    # --------------------------------------------------------
    # Real validation prompt folder
    # --------------------------------------------------------
    parser.add_argument(
        "--source_frame_path",
        type=str,
        required=True,
        help=(
            "Frame folder containing "
            "results/rank*_interval*/frame_*.prompt"
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "../runs/"
            "f2z_tcn_attack_sweep"
        ),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=8808,
    )

    # --------------------------------------------------------
    # Main controllable security variable
    # --------------------------------------------------------
    parser.add_argument(
        "--leak_counts",
        type=str,
        default=(
            "0,25,50,100,500,1000"
        ),
        help=(
            "Number of leaked FULL 8808-D f-z samples."
        ),
    )

    # --------------------------------------------------------
    # Physical response
    # --------------------------------------------------------
    parser.add_argument(
        "--data_root",
        type=str,
        default=(
            "../data/measured"
        ),
    )

    parser.add_argument(
        "--freq",
        type=str,
        default="70MHz",
    )

    # Physical state used for attack training and same-state testing.
    parser.add_argument(
        "--ours_train_temp",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--ours_train_rep",
        type=int,
        default=2,
    )

    # Different target state used only for cross-state testing.
    parser.add_argument(
        "--ours_diff_temp",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--ours_diff_rep",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
    )

    # --------------------------------------------------------
    # Baseline: selectable constellation size
    # --------------------------------------------------------
    parser.add_argument(
        "--baseline_qam_order",
        type=int,
        choices=(16, 256),
        default=16,
        help=(
            "Choose adapted 16-QAM or 256-QAM "
            "constellation transform."
        ),
    )

    parser.add_argument(
        "--baseline_rotation_deg",
        type=float,
        default=27.0,
    )

    parser.add_argument(
        "--baseline_reflection",
        action="store_true",
    )

    # --------------------------------------------------------
    # Gated Pointwise Main Branch + Residual Deep TCN attacker
    # --------------------------------------------------------
    parser.add_argument(
        "--channels",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_blocks",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--num_groups",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max_dilation_power",
        type=int,
        default=11,
        help=(
            "11 gives dilations up to 2048. "
            "With 12 blocks the theoretical receptive "
            "field exceeds 8808."
        ),
    )

    parser.add_argument(
        "--pointwise_aux_weight",
        type=float,
        default=1.0,
        help=(
            "Auxiliary loss weight forcing the pointwise branch "
            "to learn any symbol-wise f_i -> z_i mapping."
        ),
    )

    parser.add_argument(
        "--context_reg_weight",
        type=float,
        default=1e-3,
        help=(
            "Small L2 penalty on the TCN residual. It discourages "
            "unnecessary contextual corrections on simple baselines."
        ),
    )

    parser.add_argument(
        "--initial_context_logit",
        type=float,
        default=-3.0,
        help=(
            "Initial logit for the residual gate alpha=sigmoid(logit). "
            "-3 corresponds to alpha about 0.047."
        ),
    )

    # --------------------------------------------------------
    # Epoch-based optimization
    # --------------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help=(
            "Minimum number of complete passes over the leaked f-z dataset."
        ),
    )

    parser.add_argument(
        "--min_optimizer_steps",
        type=int,
        default=1000,
        help=(
            "Minimum optimizer updates for every N>0. "
            "Small-N runs automatically use more epochs."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--log_batch_every",
        type=int,
        default=100,
        help=(
            "Print one mini-batch status every this "
            "many batches; 0 disables batch logging."
        ),
    )

    # --------------------------------------------------------
    # Large-dataset generation
    # --------------------------------------------------------
    parser.add_argument(
        "--data_chunk_size",
        type=int,
        default=256,
        help=(
            "Chunk size when generating huge random "
            "N x 8808 master datasets."
        ),
    )

    # --------------------------------------------------------
    # Devices / reproducibility
    # --------------------------------------------------------
    parser.add_argument(
        "--attack_device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--save_device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--force_retrain",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------
    leak_counts = parse_int_list(
        args.leak_counts
    )

    max_leak = max(
        leak_counts
    )

    if args.sequence_length != 8808:
        raise ValueError(
            "Current prompt format expects "
            "sequence_length=8808."
        )

    if args.sequence_length % 2 != 0:
        raise ValueError(
            "sequence_length must be even."
        )

    if args.kernel_size % 2 == 0:
        raise ValueError(
            "--kernel_size must be odd so that "
            "Conv1d preserves sequence length."
        )

    if args.channels % args.num_groups != 0:
        raise ValueError(
            "--channels must be divisible by "
            "--num_groups."
        )

    if (
        "cuda" in args.attack_device
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable for attack model."
        )

    if (
        "cuda" in args.save_device
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable for prompt saving."
        )

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Verify real prompt files
    # --------------------------------------------------------
    source_prompt_dir = prompt_dir(
        args.source_frame_path,
        args.rank,
        args.interval,
    )

    real_prompt_files = sorted(
        source_prompt_dir.glob(
            "frame_*.prompt"
        )
    )

    if not real_prompt_files:
        raise FileNotFoundError(
            f"No frame_*.prompt in "
            f"{source_prompt_dir}"
        )

    first_z, _ = load_prompt_bytes(
        str(
            real_prompt_files[0]
        )
    )

    if first_z.size != args.sequence_length:
        raise ValueError(
            f"Real prompt length={first_z.size}, "
            f"expected={args.sequence_length}"
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------
    output_root = Path(
        args.output_root
    )

    model_root = (
        output_root
        / "_models"
    )

    table_root = (
        output_root
        / "_physical_tables"
    )

    baseline_root = (
        output_root
        / "_baseline_key"
    )

    training_root = (
        output_root
        / "_random_training"
    )

    prompt_output_root = (
        output_root
        / "prompt_outputs"
    )

    for directory in (
        model_root,
        table_root,
        baseline_root,
        training_root,
        prompt_output_root,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Build/load physical lookup tables
    # --------------------------------------------------------
    same_mat = get_mat_path(
        args.data_root,
        args.freq,
        args.ours_train_temp,
        args.ours_train_rep,
    )

    diff_mat = get_mat_path(
        args.data_root,
        args.freq,
        args.ours_diff_temp,
        args.ours_diff_rep,
    )

    print(
        "Attack training / same-state test:",
        same_mat,
    )

    print(
        "Different-state test:",
        diff_mat,
    )

    same_table_path = (
        table_root
        / (
            f"{args.freq}_"
            f"{args.ours_train_temp}degR"
            f"{args.ours_train_rep}_"
            f"gain{args.gain}.npz"
        )
    )

    diff_table_path = (
        table_root
        / (
            f"{args.freq}_"
            f"{args.ours_diff_temp}degR"
            f"{args.ours_diff_rep}_"
            f"gain{args.gain}.npz"
        )
    )

    same_selected = ensure_physical_table(
        same_mat,
        str(same_table_path),
        args.gain,
    )

    diff_selected = ensure_physical_table(
        diff_mat,
        str(diff_table_path),
        args.gain,
    )

    collision_statistics(
        same_selected,
        name=(
            f"{args.ours_train_temp}degR"
            f"{args.ours_train_rep}"
        ),
    )

    collision_statistics(
        diff_selected,
        name=(
            f"{args.ours_diff_temp}degR"
            f"{args.ours_diff_rep}"
        ),
    )

    # --------------------------------------------------------
    # Build baseline constellation transform
    # --------------------------------------------------------
    (
        baseline_permutation,
        baseline_inverse,
        qam_points,
    ) = build_constellation_permutation(
        M=args.baseline_qam_order,
        rotation_deg=args.baseline_rotation_deg,
        reflection=args.baseline_reflection,
    )

    baseline_key_path = (
        baseline_root
        / (
            f"qam{args.baseline_qam_order}_"
            f"rot{args.baseline_rotation_deg:g}_"
            f"reflect"
            f"{int(args.baseline_reflection)}"
            f".npz"
        )
    )

    np.savez_compressed(
        baseline_key_path,
        permutation=(
            baseline_permutation.astype(
                np.int16
            )
        ),
        inverse_permutation=(
            baseline_inverse.astype(
                np.int16
            )
        ),
        constellation_points=(
            qam_points.astype(
                np.float32
            )
        ),
        qam_order=np.asarray(
            args.baseline_qam_order
        ),
        rotation_deg=np.asarray(
            args.baseline_rotation_deg
        ),
        reflection=np.asarray(
            int(
                args.baseline_reflection
            )
        ),
    )

    print(
        f"[baseline key] saved: "
        f"{baseline_key_path}"
    )

    # --------------------------------------------------------
    # Generate/load nested master random prompts
    # --------------------------------------------------------
    if max_leak > 0:
        z_master_path = (
            training_root
            / (
                f"master_z_"
                f"N{max_leak}_"
                f"L{args.sequence_length}_"
                f"seed{args.seed}.npy"
            )
        )

        z_master = ensure_master_random_z(
            path=str(
                z_master_path
            ),
            num_samples=max_leak,
            sequence_length=args.sequence_length,
            seed=args.seed,
            chunk_size=args.data_chunk_size,
        )

        # Baseline master f.
        baseline_f_master_path = (
            training_root
            / (
                f"master_f_"
                f"baseline_qam"
                f"{args.baseline_qam_order}_"
                f"rot"
                f"{args.baseline_rotation_deg:g}_"
                f"N{max_leak}.npy"
            )
        )

        baseline_f_master = ensure_encoded_master(
            output_path=str(
                baseline_f_master_path
            ),
            z_master=z_master,
            encoder_fn=(
                lambda z_chunk:
                baseline_encode_numpy(
                    z_chunk,
                    qam_order=(
                        args.baseline_qam_order
                    ),
                    permutation=(
                        baseline_permutation
                    ),
                )
            ),
            num_samples=max_leak,
            sequence_length=args.sequence_length,
            chunk_size=args.data_chunk_size,
        )

        # Physical s1 master f.
        ours_f_master_path = (
            training_root
            / (
                f"master_f_ours_"
                f"{args.freq}_"
                f"{args.ours_train_temp}degR"
                f"{args.ours_train_rep}_"
                f"gain{args.gain}_"
                f"N{max_leak}.npy"
            )
        )

        ours_f_master = ensure_encoded_master(
            output_path=str(
                ours_f_master_path
            ),
            z_master=z_master,
            encoder_fn=(
                lambda z_chunk:
                physical_encode_numpy(
                    z_chunk,
                    same_selected,
                )
            ),
            num_samples=max_leak,
            sequence_length=args.sequence_length,
            chunk_size=args.data_chunk_size,
        )

    else:
        z_master = None
        baseline_f_master = None
        ours_f_master = None

    # --------------------------------------------------------
    # Sweep leaked f-z sample count
    # --------------------------------------------------------
    all_detail_rows = []

    for n_leak in leak_counts:
        print("\n" + "#" * 84)
        print(
            f"LEAKED FULL f-z SAMPLES: "
            f"N={n_leak}"
        )
        print("#" * 84)

        if n_leak == 0:
            baseline_model = None
            ours_model = None

        else:
            # A different initialization per N but identical architecture.
            seed_n = (
                args.seed
                + n_leak
            )

            baseline_model = train_attack_by_epochs(
                experiment_name=(
                    f"baseline_qam"
                    f"{args.baseline_qam_order}"
                ),
                f_train_u8=baseline_f_master,
                z_train_u8=z_master,
                n_leak=n_leak,
                model_path=str(
                    model_root
                    / (
                        f"baseline_qam"
                        f"{args.baseline_qam_order}_"
                        f"gated_pointwise_tcn"
                    )
                    / (
                        f"N{n_leak:06d}.pth"
                    )
                ),
                sequence_length=args.sequence_length,
                device=args.attack_device,
                epochs=args.epochs,
                min_optimizer_steps=args.min_optimizer_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                channels=args.channels,
                num_blocks=args.num_blocks,
                kernel_size=args.kernel_size,
                dropout=args.dropout,
                num_groups=args.num_groups,
                max_dilation_power=(
                    args.max_dilation_power
                ),
                pointwise_aux_weight=(
                    args.pointwise_aux_weight
                ),
                context_reg_weight=(
                    args.context_reg_weight
                ),
                initial_context_logit=(
                    args.initial_context_logit
                ),
                seed=seed_n,
                log_batch_every=(
                    args.log_batch_every
                ),
                force_retrain=(
                    args.force_retrain
                ),
            )

            ours_model = train_attack_by_epochs(
                experiment_name=(
                    f"ours_"
                    f"{args.ours_train_temp}degR"
                    f"{args.ours_train_rep}"
                ),
                f_train_u8=ours_f_master,
                z_train_u8=z_master,
                n_leak=n_leak,
                model_path=str(
                    model_root
                    / (
                        f"ours_"
                        f"{args.ours_train_temp}degR"
                        f"{args.ours_train_rep}_"
                        f"gated_pointwise_tcn"
                    )
                    / (
                        f"N{n_leak:06d}.pth"
                    )
                ),
                sequence_length=args.sequence_length,
                device=args.attack_device,
                epochs=args.epochs,
                min_optimizer_steps=args.min_optimizer_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                channels=args.channels,
                num_blocks=args.num_blocks,
                kernel_size=args.kernel_size,
                dropout=args.dropout,
                num_groups=args.num_groups,
                max_dilation_power=(
                    args.max_dilation_power
                ),
                pointwise_aux_weight=(
                    args.pointwise_aux_weight
                ),
                context_reg_weight=(
                    args.context_reg_weight
                ),
                initial_context_logit=(
                    args.initial_context_logit
                ),
                seed=seed_n,
                log_batch_every=(
                    args.log_batch_every
                ),
                force_retrain=(
                    args.force_retrain
                ),
            )

        output_folders = create_output_folders(
            source_frame_path=(
                args.source_frame_path
            ),
            output_root=str(
                prompt_output_root
            ),
            n_leak=n_leak,
            rank=args.rank,
            interval=args.interval,
            overwrite=args.overwrite,
        )

        rows = evaluate_real_prompts(
            source_frame_path=(
                args.source_frame_path
            ),
            output_folders=output_folders,
            rank=args.rank,
            interval=args.interval,
            save_device=args.save_device,
            sequence_length=(
                args.sequence_length
            ),
            n_leak=n_leak,
            qam_order=(
                args.baseline_qam_order
            ),
            baseline_permutation=(
                baseline_permutation
            ),
            same_selected=(
                same_selected
            ),
            diff_selected=(
                diff_selected
            ),
            baseline_model=(
                baseline_model
            ),
            ours_model=(
                ours_model
            ),
            attack_device=(
                args.attack_device
            ),
        )

        all_detail_rows.extend(
            rows
        )

        del baseline_model
        del ours_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Save prompt-domain summaries
    # --------------------------------------------------------
    detail_csv = (
        output_root
        / "prompt_attack_detail.csv"
    )

    write_csv(
        path=str(
            detail_csv
        ),
        rows=all_detail_rows,
        fieldnames=[
            "n_leak",
            "prompt_file",
            "condition",
            "prompt_mae8",
            "prompt_mse_norm",
        ],
    )

    summary_rows = aggregate_rows(
        all_detail_rows
    )

    summary_csv = (
        output_root
        / "prompt_attack_summary.csv"
    )

    write_csv(
        path=str(
            summary_csv
        ),
        rows=summary_rows,
        fieldnames=[
            "n_leak",
            "condition",
            "num_prompts",
            "mean_prompt_mae8",
            "mean_prompt_mse_norm",
        ],
    )

    config = {
        "source_frame_path": str(
            Path(
                args.source_frame_path
            ).resolve()
        ),
        "sequence_length": (
            args.sequence_length
        ),
        "leak_counts": (
            leak_counts
        ),
        "n0_behavior": (
            "No attack network; encoded f is "
            "directly written as prompt."
        ),
        "baseline": {
            "qam_order": (
                args.baseline_qam_order
            ),
            "rotation_deg": (
                args.baseline_rotation_deg
            ),
            "reflection": bool(
                args.baseline_reflection
            ),
            "key_path": str(
                baseline_key_path
            ),
        },
        "physical_attack_training_state": {
            "frequency": args.freq,
            "temperature": (
                args.ours_train_temp
            ),
            "repetition": (
                args.ours_train_rep
            ),
        },
        "physical_different_test_state": {
            "frequency": args.freq,
            "temperature": (
                args.ours_diff_temp
            ),
            "repetition": (
                args.ours_diff_rep
            ),
        },
        "attacker": {
            "architecture": (
                "pointwise main branch + "
                "gated deep dilated residual TCN"
            ),
            "pointwise_aux_weight": (
                args.pointwise_aux_weight
            ),
            "context_reg_weight": (
                args.context_reg_weight
            ),
            "initial_context_logit": (
                args.initial_context_logit
            ),
            "input": (
                "complete 8808-D f"
            ),
            "output": (
                "complete 8808-D z"
            ),
            "channels": args.channels,
            "num_blocks": (
                args.num_blocks
            ),
            "kernel_size": (
                args.kernel_size
            ),
            "max_dilation_power": (
                args.max_dilation_power
            ),
            "epochs": args.epochs,
            "min_optimizer_steps": (
                args.min_optimizer_steps
            ),
            "batch_size": (
                args.batch_size
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "loss": "MSELoss",
            "output_activation": (
                "none during training"
            ),
        },
        "detail_csv": str(
            detail_csv
        ),
        "summary_csv": str(
            summary_csv
        ),
    }

    config_json = (
        output_root
        / "experiment_config.json"
    )

    with open(
        config_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 84)
    print("DONE")
    print("=" * 84)
    print(
        "Prompt outputs :",
        prompt_output_root,
    )
    print(
        "Detail CSV     :",
        detail_csv,
    )
    print(
        "Summary CSV    :",
        summary_csv,
    )
    print(
        "Config         :",
        config_json,
    )

    print(
        "\nAt N=0, each output .prompt "
        "contains the encoded f directly."
    )

    print(
        "\nExample generation command:"
    )

    example_folder = (
        prompt_output_root
        / (
            f"N{leak_counts[-1]:06d}"
        )
        / "ours_diff_state_attack"
    )

    print(
        f'python generation.py '
        f'-frame_path "{example_folder}" '
        f'-rank {args.rank} '
        f'-interval {args.interval}'
    )


if __name__ == "__main__":
    main()
