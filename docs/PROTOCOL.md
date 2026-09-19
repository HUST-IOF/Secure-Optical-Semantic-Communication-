# Implemented protocol and interpretation

1. Extract seven Gaussian-fit negative-pulse features from each 800-sample waveform. An acquisition contains 256 symbols.
2. Fit a mean, a 7-to-2 PCA basis and per-coordinate extrema on the sender calibration. Preserve these parameters for receiver projection.
3. Enumerate 256 squared ordered frequency pairs. For each target byte pair, select a nearest clipped difference in the normalized response plane. Mapping collisions and quantization error are possible.
4. In the image inversion scripts, optimize using the sender calibration only. Receiver measurements enter subsequent prompt materialization, not the sender image loss.
5. Online recovery projects measured seven-dimensional responses with the stored parameters and differences adjacent responses. `recover_responses` does not accept frequency indices. `replay_bytes` explicitly simulates this using measured codebook rows.
6. The serialized rank-8 image prompt contains U of shape 77 x 8 and V of shape 8 x 1024: 8,808 byte values. Scale and zero-point fields are additional side information. The standalone codec operates on the byte payload; the research scripts retain the six-field `.prompt` interface.

The public default gain is 1.0 in the standalone codec. Some original encoder CLI defaults are 4.0; the documented research commands explicitly select 1.0. Do not mix cached tables across calibration states or gain settings.

## Attack experiment

`attack_gated_tcn.py` trains a pointwise network plus a gated dilated TCN residual. N is the number of complete 8,808-dimensional leaked `(f,z)` sequences. For cross-state evaluation, `evaluate_real_prompts` generates `f_diff` using `diff_selected` and feeds it to the same physical attacker trained under the original state. It does not train a new model for the target state.

The random training prompts and synthetic smoke fixtures are not natural-image source prompts. The research image test prompts require separate image inversion. N=0 is direct decoding of the encoded values, not an optimal zero-leakage attack. Quantization side information is retained by the prompt serialization workflow.

## What is not established

- A fixed physical mapping is learnable. This repository is not a standard cryptographic implementation or proof of semantic security.
- A state-switch attack comparison still needs a symmetric digital key-switching control and stronger semantic leakage measurements.
- The 1.4 ms historical timing excludes image generation, file I/O and offline calibration. It is not end-to-end latency. Historical timing variance is retained, not concealed.
- The measured demo keeps one encoded sequence and changes its receive responses. This is response mismatch, distinct from re-encoding under a new state for neural attack transfer.
- Full paper reproduction also requires source images, model weights, full acquisition conditions and experimental logs not included here.

