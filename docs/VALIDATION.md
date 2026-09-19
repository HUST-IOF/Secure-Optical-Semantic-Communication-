# Release validation

Validation performed on 2026-09-19. No complete image inversion or paper-scale
attack training was launched, and no remote environment was modified.

## Completed

- Local Windows / Python 3.12: **11 tests passed, 1 CUDA test skipped**.
- Dependencies used for local checks: NumPy 2.5.3, SciPy 1.18.1,
  pytest 9.1.1 and PyTorch 2.14.0+cpu.
- The standalone codec was compared numerically against functions extracted
  from the original physical encoder, including nearest-neighbour distances
  for all 65,536 target byte pairs.
- The measured example processed all three Data acquisitions, encoded 8,808
  deterministic random byte values and recovered them through measured tables.
- PyTorch CPU recovery matched NumPy byte-for-byte on 8,808 symbols. Batched
  recovery and finite gradients were checked in the tests.
- The actual attack CLI completed N=0 and N=2 with two residual blocks,
  eight channels and one optimizer step for each trained model. Six output
  prompts were checked for field names, shapes and unchanged quantization
  side information. These synthetic fixtures do not yield image-quality claims.
- The original receiver timing script completed a small execution check with
  the included Data and Dict files; this is not a reproduction of paper timing.
- The Python package built successfully as a wheel. Release sample checksums,
  Python syntax and the small-file/private-path checks passed. Build outputs,
  downloaded dependencies and generated experiments are excluded from the ZIP.

## Measured codec example

Seed 2026, gain 1.0, 8,808 random byte values, 4,404 frequency pairs:

| Receive condition | Byte MAE |
|---|---:|
| Same calibration acquisition | 1.086739 |
| Same nominal temperature, independent repeat | 1.969233 |
| Different-temperature response mismatch | 88.149523 |

These values characterize this deterministic codebook-replay example only.
They are not PSNR, LPIPS or neural attack-transfer results.

## CUDA status

The inspected remote Python environment reported `torch.cuda.is_available()`
as false. The local check used a CPU PyTorch build. Consequently, the CUDA
execution test could not run. CUDA code paths and an automatic CUDA parity /
gradient test are included, but hardware execution is **not verified** in this
release. Run `python examples/torch_receiver_demo.py --device cuda`,
`python examples/attack_smoke.py --device cuda` and `python -m pytest -q` on a
CUDA-enabled environment to perform those checks.

## Original image and timing code

The CUDA-only SD-Turbo image pipeline was retained, not rerun. Portable
CPU-saved prompt tensors are now loaded onto CUDA explicitly by generation.
It still requires two adjacent prompts and the inversion `init.pth` state.

The original timing script is a historical compute proxy: fixed peak samples
and normalization to [0,1], without the full Gaussian-fit extraction,
[-1,1] normalization, clipping and byte conversion in the standalone codec.
Its automatic peak fallback selected adjacent sample positions on the supplied
dictionary in the short check. Peak calibration and matching alignment require
review before interpreting its timings as a validated semantic receiver.
Historical CSVs have been preserved rather than replaced by new timings.

The full model dependency stack, checkpoint acquisition, all image metrics,
paper-scale attacks, power sweep and remote GPU hardware remain outside the
completed validation. See `PROTOCOL.md` for further experimental boundaries.
