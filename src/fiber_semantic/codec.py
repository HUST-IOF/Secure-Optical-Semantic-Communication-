"""CPU implementation of the experiment's PCA/difference physical codec.

No image model is needed. ``replay_bytes`` is explicitly offline table replay;
the online API ``recover_responses`` accepts measured responses, not indices.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def _responses(value, *, calibration=False):
    a = np.asarray(value, dtype=np.float64)
    if calibration and a.shape == (7, 256):
        a = a.T
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
        raise ValueError("Responses must be a finite N x 7 array.")
    if calibration and a.shape != (256, 7):
        raise ValueError("Calibration requires 256 x 7 responses.")
    return a


@dataclass(frozen=True)
class Calibration:
    mean: np.ndarray
    basis: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    singular_values: np.ndarray
    gain: float = 1.0
    eps: float = 1e-12

    def __post_init__(self):
        for name, shape in [('mean', (1, 7)), ('basis', (7, 2)),
                            ('lower', (1, 2)), ('upper', (1, 2)),
                            ('singular_values', (7,))]:
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid calibration field: {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be finite and positive")
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if np.any(self.upper <= self.lower):
            raise ValueError("Calibration must span both PCA coordinates")

    @classmethod
    def fit(cls, responses, gain=1.0):
        """Fit once on the sender calibration; never refit on online Rx data."""
        a = _responses(responses, calibration=True)
        mean = a.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(a - mean, full_matrices=False)
        if s[1] <= max(s[0], 1.0) * 1e-12:
            raise ValueError("At least two independent response dimensions are required")
        basis = vt[:2].T
        b = (a - mean) @ basis
        return cls(mean, basis, b.min(axis=0, keepdims=True),
                   b.max(axis=0, keepdims=True), s, float(gain))

    def project(self, responses):
        b = (_responses(responses) - self.mean) @ self.basis
        return 2 * (b - self.lower) / (self.upper - self.lower + self.eps) - 1

    def build_table(self, responses):
        """Return 65536 x 2 ordered frequency indices for all byte pairs.

        Tied nearest neighbours can differ across SciPy versions. Their
        projected distances are equivalent; indices are not guaranteed unique.
        """
        k = self.project(_responses(responses, calibration=True))
        i, j = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        pairs = np.column_stack((i.ravel(), j.ravel())).astype(np.uint16)
        candidates = np.clip(self.gain * (k[pairs[:, 0]] - k[pairs[:, 1]]), -1, 1)
        a, b = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
        target = 2 * np.column_stack((a.ravel(), b.ravel())) / 255 - 1
        _, nearest = cKDTree(candidates).query(target, k=1, workers=1)
        return pairs[nearest]

    def save(self, path):
        """Save numeric fields only; loading does not require pickle."""
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, mean_A=self.mean, V2=self.basis,
                                b_min=self.lower, b_max=self.upper,
                                singular_values=self.singular_values,
                                gain=self.gain, eps=self.eps)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(d['mean_A'], d['V2'], d['b_min'], d['b_max'],
                       d['singular_values'], float(d['gain']), float(d['eps']))


def encode_bytes(values, table):
    """Encode an even number of uint8 values. No implicit wrapping or padding."""
    z = np.asarray(values)
    if z.dtype != np.uint8 or z.size == 0 or z.size % 2:
        raise ValueError("Input must contain a nonempty, even number of uint8 values")
    t = np.asarray(table)
    if (t.shape != (65536, 2) or not np.issubdtype(t.dtype, np.integer)
            or np.any(t < 0) or np.any(t > 255)):
        raise ValueError("Table must be integer 65536 x 2 with indices in [0, 255]")
    z = z.reshape(-1, 2).astype(np.int64)
    return t[z[:, 0] * 256 + z[:, 1]].copy()


def recover_responses(responses, calibration):
    """Recover bytes from ordered N x 7 waveform features, without indices."""
    k = calibration.project(responses)
    if len(k) == 0 or len(k) % 2:
        raise ValueError("An even, nonzero number of received symbols is required")
    q = np.clip(calibration.gain * (k[0::2] - k[1::2]), -1, 1)
    return np.clip(np.rint((q + 1) * 255 / 2), 0, 255).astype(np.uint8).ravel()


def replay_bytes(frequency_pairs, receive_codebook, calibration):
    """Offline replay using measured Rx codebook rows. Not live transmission."""
    pairs = np.asarray(frequency_pairs)
    if (pairs.ndim != 2 or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0) or np.any(pairs > 255)):
        raise ValueError("Expected integer N x 2 frequency indices in [0, 255]")
    responses = _responses(receive_codebook, calibration=True)
    return recover_responses(responses[pairs.ravel()], calibration)

