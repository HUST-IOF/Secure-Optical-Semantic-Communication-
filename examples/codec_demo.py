"""Run from the repository root after `python -m pip install -e .`."""
import argparse
import json
from pathlib import Path
import numpy as np
from fiber_semantic import Calibration, encode_bytes, replay_bytes
from fiber_semantic.waveforms import load_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measured', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/codec_demo.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    if args.measured:
        data = Path(__file__).resolve().parents[1] / 'data/measured'
        tx = load_features(data / 'Data20260508_20km_70MHz_50deg_1.mat')
        rx = load_features(data / 'Data20260508_20km_70MHz_50deg_2.mat')
        mismatch = load_features(data / 'Data20260508_20km_70MHz_48deg_2.mat')
    else:
        latent = rng.normal(size=(256, 2))
        mixing = rng.normal(size=(2, 7))
        tx = latent @ mixing + 3
        rx = tx + rng.normal(scale=.01, size=tx.shape)
        mismatch = tx[rng.permutation(256)]
    calibration = Calibration.fit(tx, gain=1.0)
    table = calibration.build_table(tx)
    prompt = rng.integers(0, 256, 8808, dtype=np.uint8)
    pairs = encode_bytes(prompt, table)
    reports = {}
    for name, response in [('same_calibration', tx), ('repeated_measurement', rx),
                           ('mismatched_response', mismatch)]:
        recovered = replay_bytes(pairs, response, calibration)
        reports[name] = {'mae_byte': float(np.abs(recovered.astype(float)-prompt).mean())}
    report = {'mode': 'measured_codebook_replay' if args.measured else 'synthetic',
              'seed': 2026, 'gain': 1.0, 'prompt_bytes': 8808,
              'frequency_pairs': int(len(pairs)), 'results': reports,
              'note': 'Random byte prompt; no image decoder or neural attack. '
                      'Response mismatch is not an attack-transfer experiment.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

