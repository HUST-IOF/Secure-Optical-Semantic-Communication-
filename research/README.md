# Original research workflows

The original code is retained with limited portability edits. Source hashes and
changes are listed in `../docs/source_manifest.json`. In particular,
`secure_inversion_txloss.py` comes from the actual file named
`secure_inversion_diff.py`; its loss uses the sender matrix. The historically
mentioned `secure_inversion_txloss_multifreq.py` was not present in the inspected
project root, so no unverifiable version is claimed.

## 1. Minimal attack execution, CPU or GPU

From the repository root, install PyTorch and run:

```bash
python -m pip install -e ".[attack]"
python examples/attack_smoke.py --device auto
```

This builds tables from the public measured states, creates one synthetic
rank-8 prompt and executes N=0 and N=2 cases with one optimizer step per model.
It checks that all six output prompts preserve the expected fields and shapes.
It is an execution test, not a trained attack-quality result.

## 2. Image inversion and generation (CUDA)

Use a separate Linux/CUDA environment for the legacy image pipeline. The
original Python 3.10 environment is recorded in `environment.reference.yml`,
with its private machine prefix removed. It contains historical packages and
platform build pins and is **not** a new tested lock file. The standalone CPU
environment does not install this full stack.

The original source used torch 2.5.1, diffusers 0.26.1,
huggingface_hub 0.25.0, transformers 4.37.2, torchmetrics 1.3.0.post0,
omegaconf 2.3.0, einops 0.7.0, open-clip-torch 2.24.0,
pytorch-lightning 2.0.1, kornia 0.6.9 and streamlit 1.31.0. Other imports include
OpenCV, Pillow, safetensors, invisible-watermark and openai-clip. Retain a
working original environment when reproducing the original image pipeline;
modern dependency compatibility has not been established by the CPU tests.

Obtain SD-Turbo weights from its official model distribution under its terms,
and place `sd_turbo.safetensors` in `research/checkpoints/`. The code also loads
`madebyollin/taesd`, CLIP/filter assets and LPIPS assets, which may download on
first use. Large weights are not bundled. Small upstream filter heads are
retained in `scripts/util/detection/`.

Prepare at least two source PNGs as `00000.png`, `00001.png`, etc. The retained
generation loop processes adjacent prompt pairs and also requires `init.pth`
from inversion. A single prompt alone does not produce images in that loop.
Use an isolated
input/output folder because the historical workflow writes prompt results.
From `research/`, with source images in `../runs/source_images/`:

```bash
python secure_inversion_txloss.py -frame_path ../runs/source_images -max_id 1 -rank 8 -interval 1 --secure_sweep --data_root ../data/measured --output_root ../runs/inversion_demo --freq 70MHz --temps 48,50 --trans_temp 50 --trans_rep 1 --recv_rep 2 --gain 1.0
```

The sender optimization runs once per selected frequency; receiver conditions
are materialized afterwards. The example uses only included acquisitions.
No long inversion is started by the quick-start tests.

Generate images for a resulting condition folder:

```bash
python generation.py -frame_path ../runs/inversion_demo/70MHz_Tx50degR1_Rx50degR2_gain1.0 -rank 8 -interval 1
```

The retained power sweep script requires additional power acquisitions that
are not in the selected sample. Supply their directory explicitly and inspect
its command-line arguments before launching the full sweep.

## 3. Full attack workflow

Replace the source folder below with an image-derived prompt folder. The
directory must contain `results/rank8_interval1/frame_*.prompt` and source
images if later image evaluation is required. From `research/`:

```bash
python attack_gated_tcn.py --source_frame_path ../runs/source_prompts --output_root ../runs/attack --data_root ../data/measured --freq 70MHz --ours_train_temp 50 --ours_train_rep 2 --ours_diff_temp 48 --ours_diff_rep 2 --gain 1.0 --leak_counts 0,25,50,100,250,500,1000,2000,4000,16000 --attack_device cuda:0 --save_device cpu
```

Use `--attack_device cpu` for CPU operation. The default full attack can be
expensive. Save tensors on CPU for portable files; generation moves tensors
to its CUDA device. Only load trusted `.prompt`/checkpoint files: the original
research serialization uses PyTorch pickle loading.

```bash
python run_generation_and_eval_attack.py --attack_root ../runs/attack --generation_script generation.py --leak_counts 0,25,50,100,250,500,1000,2000,4000,16000 --output_csv ../runs/attack_generation_summary.csv --detail_csv ../runs/attack_generation_detail.csv
```

This evaluator originally had a hardcoded N=250. The release exposes
`--leak_counts`; missing condition folders are reported and skipped.

## 4. Receiver runtime example

From the repository root:

```bash
python research/receiver_runtime_comparison.py --data_mat data/measured/Data20260508_20km_70MHz_50deg_2.mat --dict_mat data/measured/Dict20260508_20km_70MHz_50deg_1.mat --save_dir runs/runtime_demo --warmup 3 --repeat_codeword 10 --repeat_image 3
```

This short run checks execution only. For timing studies, increase repetitions,
record hardware/library/thread settings and inspect the raw distributions.
Offline calibration and image generation are excluded. The NumPy timing
implementation measures CPU processing and does not benchmark `TorchReceiver`.

## Results and status

Historical CSVs in `../data/` remain separate from newly generated `../runs/`.
Read `../docs/VALIDATION.md` for what was actually checked during release
preparation. Source inclusion alone is not a claim that full image inversion,
all attack training or all paper plots were rerun.
