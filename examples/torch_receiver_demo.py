"""Optional direct receiver demo on CPU or CUDA. No model weights needed."""
import argparse
import json
import numpy as np
import torch
from fiber_semantic import Calibration, recover_responses
from fiber_semantic.torch_receiver import TorchReceiver

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
args = parser.parse_args()
device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
if device == 'cuda' and not torch.cuda.is_available():
    parser.error('CUDA was requested but is unavailable; install a CUDA-enabled PyTorch build or use --device cpu')
rng = np.random.default_rng(2026)
calibration = Calibration.fit(rng.normal(size=(256, 7)))
responses = rng.normal(size=(8808, 7))
receiver = TorchReceiver(calibration).to(device)
tensor = torch.tensor(responses, dtype=torch.float64, device=device)
actual = receiver.recover_bytes(tensor).cpu().numpy()
expected = recover_responses(responses, calibration)
np.testing.assert_array_equal(actual, expected)
print(json.dumps({'device':device, 'symbols':8808, 'matches_numpy':True,
                  'note':'Functional parity test; not an end-to-end latency benchmark.'}))

