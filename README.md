# Secure optical semantic communication

Research code for **Secure optical semantic communication through reconfigurable fiber responses**.

This repository provides a compact, runnable implementation of the physical response codec, measured waveform examples, and the research scripts for image inversion, neural attacks and receiver timing. It is a selected code release, not a complete archive of all paper experiments.

## What the method does

An offline calibration maps 256 optical frequencies to seven measured responses. A fixed PCA projection maps responses to two coordinates. The transmitter selects ordered frequency pairs; the receiver projects their measured responses and takes their difference to recover the semantic representation. A frozen generative decoder reconstructs images from the recovered low-rank prompt.

```text
image -> state-constrained prompt optimization -> frequency pairs
      -> optical link / MMF response -> seven waveform features
      -> fixed projection and difference -> prompt -> image decoder
```

The receiver does not need to identify frequency symbols. The sender optimization uses only the sender calibration. Changing a physical state changes the mapping; a fixed mapping can still be learned by an attacker. See [protocol and limitations](docs/PROTOCOL.md).

## Quick start: CPU, no model download

Use Python 3.10 or newer in an isolated environment. From the repository root:

```bash
python -m pip install -e ".[test]"
python examples/codec_demo.py
python examples/codec_demo.py --measured
python -m pytest -q
```

The measured demo uses three included acquisitions: 50 degrees C repeat 1 for calibration, 50 degrees C repeat 2 for repeated measurement, and 48 degrees C repeat 2 for a response-mismatch example. It reconstructs a deterministic random byte prompt and reports byte MAE. It does **not** run an image model or reproduce attack-transfer metrics.

## CPU and GPU support

| Component | CPU | CUDA GPU |
|---|---|---|
| Offline waveform extraction, PCA and lookup construction | Yes | CPU preprocessing |
| Direct semantic recovery (`TorchReceiver`) | Yes | Yes |
| Gated pointwise/TCN attack | Yes | Yes, `--attack_device cuda:0` |
| Original image inversion and generation | Not supported by the retained scripts | Required |

Install PyTorch for your operating system and CUDA driver using the [official installation selector](https://pytorch.org/get-started/locally/), then install this project's optional components:

```bash
python -m pip install -e ".[test,gpu]"
python examples/torch_receiver_demo.py --device auto
python examples/torch_receiver_demo.py --device cuda
python examples/attack_smoke.py --device auto
```

`--device cuda` fails clearly if CUDA is unavailable; `auto` selects CUDA when available and otherwise uses CPU. Dependencies and model weights are installed separately and are not included in the repository. CUDA support does not imply a speedup for every workload.

## Research workflows

See [research instructions](research/README.md) for:

- image-to-frequency inversion and subsequent receiver materialization;
- same-state and different-state neural attacks;
- image generation and PSNR/LPIPS evaluation;
- receiver timing from the supplied waveform and dictionary.

The image workflow needs an NVIDIA GPU, additional dependencies and separately obtained pretrained weights. The source environment is provided as a historical reference; it is not the lightweight CPU test environment.

## Repository contents

```text
src/fiber_semantic/    portable codec, waveform extraction, PyTorch receiver
examples/             synthetic, measured and CPU/CUDA smoke examples
tests/                original-code regression and functional tests
research/             selected original experiment scripts and dependencies
data/measured/        four small MATLAB acquisitions and provenance
data/published_metrics/ existing aggregate results, not regenerated results
data/historical_timing/ original timing summary and individual samples
docs/                 protocol, provenance, validation and publishing notes
tools/                release checks
```

Only the selected public sample states are included. Their responses are public test fixtures, not secret operational keys. The experiment uses a **20-km single-mode-fiber link**, as confirmed by the authors. Sample filenames and commands consistently use `20km`; see [data notes](data/measured/README.md).

## Attribution and reuse

The image workflow builds on **Promptus: Can Prompts Streaming Replace Video Streaming with Stable Diffusion** and its bundled generative-model code. The original Apache-2.0 license and notices are retained; see [NOTICE](NOTICE). Model checkpoints have their own licenses. Experimental data reuse is described in [DATA_LICENSE.md](DATA_LICENSE.md).

No journal acceptance, DOI, cryptographic security proof or end-to-end real-time claim is implied by this release.

中文说明：[README.zh-CN.md](README.zh-CN.md)。

