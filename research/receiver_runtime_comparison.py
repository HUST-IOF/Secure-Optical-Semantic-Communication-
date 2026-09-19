#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Receiver runtime comparison:
1) Correlation / fingerprint matching
2) Ours: direct physical-response-to-semantic recovery

Both methods start from the same received raw waveform.

Matching baseline:
    waveform -> Pearson correlation with all dictionary entries -> argmax index

Ours:
    waveform -> 7 calibrated peaks -> fixed PCA (7->2) -> normalization
    For a complete image, every two projected responses are differenced
    to recover two semantic values.

Excluded from timing:
    - MAT-file I/O
    - offline dictionary/PCA calibration
    - common generative decoder inference

Outputs:
    receiver_runtime_summary.csv
    receiver_runtime_details.csv
    receiver_runtime_table.tex
"""

# Stabilize CPU BLAS timing.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks


# ------------------------------------------------------------
# Data loading: reproduce MATLAB reshape(X, 800, [])'
# ------------------------------------------------------------

def load_variable(path, variable):
    obj = loadmat(path)
    if variable not in obj:
        keys = [k for k in obj if not k.startswith("__")]
        raise KeyError(
            f"{variable!r} not found in {path}. Available keys: {keys}"
        )
    return np.asarray(obj[variable])


def matlab_waveforms(x, frame_length=800):
    """
    MATLAB:
        X = reshape(X, FrameLength, []);
        X = X';

    Returns [num_codewords, frame_length].
    """
    flat = np.ravel(np.asarray(x), order="F")
    if flat.size % frame_length != 0:
        raise ValueError(
            f"{flat.size} samples cannot be divided by frame_length={frame_length}"
        )

    y = np.reshape(
        flat,
        (frame_length, -1),
        order="F",
    )
    return np.ascontiguousarray(y.T, dtype=np.float64)


# ------------------------------------------------------------
# Uploaded MATLAB matching algorithm, vectorized in NumPy
# ------------------------------------------------------------

def normalize_corr_rows(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def prepare_dictionary(dict_waveforms):
    # Offline preprocessing, not timed.
    return normalize_corr_rows(dict_waveforms)


def match_one_codeword(waveform, dict_norm):
    """
    Same decision rule as corr2 against every dictionary waveform.
    """
    q = np.asarray(waveform, dtype=np.float64)[None, :]
    qn = normalize_corr_rows(q)
    corr = qn @ dict_norm.T
    return int(np.argmax(corr[0]))


def match_image(waveforms, dict_norm):
    """
    Optimized exhaustive matching for all codewords in one image.
    Complexity: O(B * M * T)
      B: codewords/image
      M: dictionary size
      T: samples/codeword
    """
    qn = normalize_corr_rows(waveforms)
    corr = qn @ dict_norm.T
    return np.argmax(corr, axis=1)


def matlab_style_mismatch_count(dict_waveforms, data_waveforms):
    """
    Reproduce:
        MaxPair = find_max_correlation_pairs(Dict', Data')
        countUnequalRows(MaxPair)
    """
    d = normalize_corr_rows(dict_waveforms)
    x = normalize_corr_rows(data_waveforms)
    corr = d @ x.T
    best = np.argmax(corr, axis=1)

    n = min(len(best), len(dict_waveforms), len(data_waveforms))
    return int(np.sum(best[:n] != np.arange(n)))


# ------------------------------------------------------------
# Ours: direct response-to-semantic recovery
# ------------------------------------------------------------

def auto_peak_indices(dict_waveforms):
    """
    Infer 7 fixed peak positions from calibration data.
    Peak-location calibration is offline and excluded from timing.
    """
    x = np.asarray(dict_waveforms, dtype=np.float64)
    x = x - np.median(x, axis=1, keepdims=True)
    profile = np.mean(np.abs(x), axis=0)

    distance = max(1, profile.size // 10)
    peaks, props = find_peaks(
        profile,
        distance=distance,
        prominence=max(np.ptp(profile) * 0.02, 1e-12),
    )

    if len(peaks) >= 7:
        idx = np.argsort(props["prominences"])[-7:]
        return np.sort(peaks[idx]).astype(np.int64)

    # Fallback: strongest point in each of seven temporal regions.
    edges = np.linspace(0, profile.size, 8, dtype=int)
    result = []
    for i in range(7):
        a, b = edges[i], edges[i + 1]
        result.append(a + int(np.argmax(profile[a:b])))
    return np.asarray(result, dtype=np.int64)


def parse_peak_indices(text):
    if text is None or text.strip() == "":
        return None
    v = [int(x.strip()) for x in text.split(",")]
    if len(v) != 7:
        raise ValueError("--peak_indices requires exactly 7 indices")
    return np.asarray(v, dtype=np.int64)


def fit_direct_calibration(dict_waveforms, peak_indices, eps=1e-12):
    """
    Offline:
      H: [256, 7]
      PCA: 7 -> 2
      fixed min-max normalization
    """
    H = dict_waveforms[:, peak_indices]
    mu = H.mean(axis=0, keepdims=True)
    H0 = H - mu

    _, _, vt = np.linalg.svd(H0, full_matrices=False)
    V2 = vt[:2].T

    K = H0 @ V2
    kmin = K.min(axis=0, keepdims=True)
    scale = np.maximum(
        K.max(axis=0, keepdims=True) - kmin,
        eps,
    )

    return {
        "peak_indices": peak_indices,
        "mu": mu,
        "V2": V2,
        "kmin": kmin,
        "scale": scale,
    }


def ours_one_codeword(waveform, cal):
    """
    Raw waveform -> 7 peaks -> 2-D calibrated physical response.
    No dictionary matching or frequency identification.
    """
    h = waveform[cal["peak_indices"]][None, :]
    k = (h - cal["mu"]) @ cal["V2"]
    k = (k - cal["kmin"]) / cal["scale"]
    return k[0]


def ours_one_image(waveforms, cal, gain=1.0):
    """
    B raw waveforms -> B two-dimensional physical responses.
    Every two consecutive codewords form one pair:
        q = gain * (k1 - k2)

    For B=8808:
        4404 pairs * 2 values = 8808 semantic values.
    """
    h = waveforms[:, cal["peak_indices"]]
    k = (h - cal["mu"]) @ cal["V2"]
    k = (k - cal["kmin"]) / cal["scale"]

    usable = (len(k) // 2) * 2
    k = k[:usable]
    z = gain * (k[0::2] - k[1::2])
    return z.reshape(-1)


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

def benchmark(func, warmup, repeats):
    for _ in range(warmup):
        func()

    values_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        values_ms.append((t1 - t0) / 1e6)

    v = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(v.mean()),
        "std_ms": float(v.std(ddof=1) if len(v) > 1 else 0.0),
        "median_ms": float(np.median(v)),
        "all_ms": v,
    }


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

def save_outputs(save_dir, summary, details):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = save_dir / "receiver_runtime_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "single_codeword_mean_us",
                "single_codeword_std_us",
                "one_image_mean_ms",
                "one_image_std_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    details_csv = save_dir / "receiver_runtime_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "scope", "repeat", "time_ms"],
        )
        writer.writeheader()
        writer.writerows(details)

    b, o = summary
    table = rf"""\begin{{table}}[t]
    \centering
    \caption{{Receiver-side processing latency. File I/O, offline calibration,
    and the common generative decoder are excluded. Values are mean
    $\pm$ s.d.}}
    \label{{tab:receiver_latency}}
    \begin{{tabular}}{{lcc}}
        \hline
        Method & Single codeword ($\mu$s) & One image (ms) \\
        \hline
        Correlation matching &
        {b["single_codeword_mean_us"]:.3f} $\pm$ {b["single_codeword_std_us"]:.3f} &
        {b["one_image_mean_ms"]:.3f} $\pm$ {b["one_image_std_ms"]:.3f} \\
        Ours &
        {o["single_codeword_mean_us"]:.3f} $\pm$ {o["single_codeword_std_us"]:.3f} &
        {o["one_image_mean_ms"]:.3f} $\pm$ {o["one_image_std_ms"]:.3f} \\
        \hline
    \end{{tabular}}
\end{{table}}
"""
    tex_path = save_dir / "receiver_runtime_table.tex"
    tex_path.write_text(table, encoding="utf-8")

    return summary_csv, details_csv, tex_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_mat", required=True)
    p.add_argument("--dict_mat", required=True)
    p.add_argument("--data_var", default="Data")
    p.add_argument("--dict_var", default="Dict")

    p.add_argument("--frame_length", type=int, default=800)
    p.add_argument("--num_states", type=int, default=256)
    p.add_argument("--image_codewords", type=int, default=8808)

    p.add_argument(
        "--peak_indices",
        default=None,
        help="Optional seven zero-based indices, e.g. 80,170,260,350,440,530,620",
    )
    p.add_argument("--gain", type=float, default=1.0)

    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeat_codeword", type=int, default=1000)
    p.add_argument("--repeat_image", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--save_dir",
        default="./receiver_runtime_comparison",
    )

    args = p.parse_args()

    if args.image_codewords % 2 != 0:
        raise ValueError("--image_codewords must be even")

    # Load data.
    Data = load_variable(args.data_mat, args.data_var)
    Dict = load_variable(args.dict_mat, args.dict_var)

    data_w = matlab_waveforms(Data, args.frame_length)
    dict_w = matlab_waveforms(Dict, args.frame_length)

    n_states = min(args.num_states, len(dict_w))
    dict_w = np.ascontiguousarray(dict_w[:n_states])

    print("Data waveforms:", data_w.shape)
    print("Dict waveforms:", dict_w.shape)

    # Reproduce MATLAB mismatch count when dimensions permit.
    if len(data_w) >= n_states:
        mismatches = matlab_style_mismatch_count(
            dict_w,
            data_w[:n_states],
        )
        print(
            f"MATLAB-style unequal rows: {mismatches}/{n_states}"
        )

    # Offline matching dictionary.
    dict_norm = prepare_dictionary(dict_w)

    # Offline calibration for ours.
    peaks = parse_peak_indices(args.peak_indices)
    if peaks is None:
        peaks = auto_peak_indices(dict_w)

    if np.any(peaks < 0) or np.any(peaks >= args.frame_length):
        raise ValueError(f"Invalid peak indices: {peaks.tolist()}")

    cal = fit_direct_calibration(dict_w, peaks)

    print("Calibrated seven peak indices:", peaks.tolist())

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "direct_recovery_calibration.npz",
        peak_indices=peaks,
        mu=cal["mu"],
        V2=cal["V2"],
        kmin=cal["kmin"],
        scale=cal["scale"],
    )

    rng = np.random.default_rng(args.seed)

    # One measured codeword.
    single = np.ascontiguousarray(
        data_w[int(rng.integers(0, len(data_w)))]
    )

    # One-image workload. If Data has only 256 measured codewords,
    # sample them with replacement to form 8808 codewords.
    image_idx = rng.integers(
        0,
        len(data_w),
        size=args.image_codewords,
    )
    image_waveforms = np.ascontiguousarray(data_w[image_idx])

    # Timing.
    b_single = benchmark(
        lambda: match_one_codeword(single, dict_norm),
        args.warmup,
        args.repeat_codeword,
    )
    o_single = benchmark(
        lambda: ours_one_codeword(single, cal),
        args.warmup,
        args.repeat_codeword,
    )

    image_warmup = max(3, args.warmup // 5)

    b_image = benchmark(
        lambda: match_image(image_waveforms, dict_norm),
        image_warmup,
        args.repeat_image,
    )
    o_image = benchmark(
        lambda: ours_one_image(image_waveforms, cal, args.gain),
        image_warmup,
        args.repeat_image,
    )

    summary = [
        {
            "method": "Correlation matching",
            "single_codeword_mean_us": b_single["mean_ms"] * 1000,
            "single_codeword_std_us": b_single["std_ms"] * 1000,
            "one_image_mean_ms": b_image["mean_ms"],
            "one_image_std_ms": b_image["std_ms"],
        },
        {
            "method": "Ours",
            "single_codeword_mean_us": o_single["mean_ms"] * 1000,
            "single_codeword_std_us": o_single["std_ms"] * 1000,
            "one_image_mean_ms": o_image["mean_ms"],
            "one_image_std_ms": o_image["std_ms"],
        },
    ]

    details = []
    for method, scope, result in [
        ("Correlation matching", "single_codeword", b_single),
        ("Ours", "single_codeword", o_single),
        ("Correlation matching", "one_image", b_image),
        ("Ours", "one_image", o_image),
    ]:
        for i, t in enumerate(result["all_ms"]):
            details.append({
                "method": method,
                "scope": scope,
                "repeat": i,
                "time_ms": float(t),
            })

    summary_csv, details_csv, tex_path = save_outputs(
        save_dir,
        summary,
        details,
    )

    print("\n" + "=" * 84)
    print(
        f"{'Method':28s}"
        f"{'Single codeword (us)':>26s}"
        f"{'One image (ms)':>26s}"
    )
    print("-" * 84)

    for row in summary:
        a = (
            f'{row["single_codeword_mean_us"]:.3f}'
            f' ± {row["single_codeword_std_us"]:.3f}'
        )
        b = (
            f'{row["one_image_mean_ms"]:.3f}'
            f' ± {row["one_image_std_ms"]:.3f}'
        )
        print(
            f'{row["method"]:28s}{a:>26s}{b:>26s}'
        )

    print("=" * 84)

    single_speedup = (
        summary[0]["single_codeword_mean_us"]
        / summary[1]["single_codeword_mean_us"]
    )
    image_speedup = (
        summary[0]["one_image_mean_ms"]
        / summary[1]["one_image_mean_ms"]
    )

    print(f"Single-codeword speed-up: {single_speedup:.2f}x")
    print(f"One-image speed-up:       {image_speedup:.2f}x")

    print("\nSaved:")
    print(summary_csv)
    print(details_csv)
    print(tex_path)


if __name__ == "__main__":
    main()
