import numpy as np
import pytest
torch = pytest.importorskip('torch')
from fiber_semantic import Calibration, recover_responses
from fiber_semantic.torch_receiver import TorchReceiver


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_receiver_matches_cpu_and_has_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware is unavailable')
    rng = np.random.default_rng(11)
    c = Calibration.fit(rng.normal(size=(256, 7)))
    responses = rng.normal(size=(2, 40, 7))
    x = torch.tensor(responses, dtype=torch.float64, device=device, requires_grad=True)
    model = TorchReceiver(c).to(device)
    expected = np.stack([recover_responses(a,c) for a in responses])
    np.testing.assert_array_equal(model.recover_bytes(x).detach().cpu().numpy(), expected)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

