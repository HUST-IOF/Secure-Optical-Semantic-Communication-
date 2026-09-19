import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial import cKDTree
from fiber_semantic import Calibration, encode_bytes, recover_responses, replay_bytes
from fiber_semantic.waveforms import load_features

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def calibrated():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(256, 7))
    c = Calibration.fit(a)
    return a, c, c.build_table(a)


def test_matches_original_experiment(calibrated):
    """Numerical regression against original functions, without loading SD-Turbo."""
    source = (ROOT/'research/secure_inversion_txloss.py').read_text(encoding='utf-8')
    names = {'_ensure_256x7', '_build_pca2_params', '_all_2d_targets',
             '_build_single_norm_diff_clip_table'}
    module = ast.parse(source)
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    ns = {'np': np, 'cKDTree': cKDTree}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-codec>', 'exec'), ns)
    a, c, table = calibrated
    expected, _, params, _ = ns['_build_single_norm_diff_clip_table'](a, gain=1.0)
    np.testing.assert_allclose(c.mean, params['mean_A'])
    np.testing.assert_allclose(c.basis, params['V2'])
    k = c.project(a)
    actual_values = np.clip(k[table[:, 0]]-k[table[:, 1]], -1, 1)
    reference_values = np.clip(k[expected[:, 0]]-k[expected[:, 1]], -1, 1)
    # Compare distances because nearest-neighbour ties need not use identical indices.
    z = np.arange(65536)
    targets = 2*np.column_stack((z//256, z%256))/255-1
    np.testing.assert_allclose(np.linalg.norm(actual_values-targets, axis=1),
                               np.linalg.norm(reference_values-targets, axis=1), atol=1e-12)


def test_online_does_not_need_symbol_indices(calibrated):
    a, c, table = calibrated
    prompt = np.arange(256, dtype=np.uint8)
    pairs = encode_bytes(prompt, table)
    streamed_responses = a[pairs.ravel()]
    np.testing.assert_array_equal(recover_responses(streamed_responses, c),
                                   replay_bytes(pairs, a, c))


def test_save_load_preserves_fixed_coordinate_system(calibrated, tmp_path):
    a, c, _ = calibrated
    path = tmp_path/'calibration.npz'
    c.save(path)
    loaded = Calibration.load(path)
    changed_rx = a * 1.2 + 0.3
    np.testing.assert_array_equal(c.project(changed_rx), loaded.project(changed_rx))
    assert not loaded.basis.flags.writeable


@pytest.mark.parametrize('bad', [np.array([-1, 256]), np.array([1],dtype=np.uint8),
                               np.array([],dtype=np.uint8), np.array([1., 2.])])
def test_rejects_invalid_prompt(calibrated, bad):
    with pytest.raises(ValueError):
        encode_bytes(bad, calibrated[2])


def test_rejects_bad_calibration():
    with pytest.raises(ValueError):
        Calibration.fit(np.ones((256, 7)))
    with pytest.raises(ValueError):
        Calibration.fit(np.full((256, 7), np.nan))


def test_gain_survives_serialization(calibrated, tmp_path):
    a, _, _ = calibrated
    c = Calibration.fit(a, gain=4.0)
    c.save(tmp_path/'gain.npz')
    assert Calibration.load(tmp_path/'gain.npz').gain == 4.0


def test_measured_waveforms_and_fixed_rx_projection():
    d = ROOT/'data/measured'
    tx = load_features(d/'Data20260508_20km_70MHz_50deg_1.mat')
    rx = load_features(d/'Data20260508_20km_70MHz_50deg_2.mat')
    assert tx.shape == rx.shape == (256, 7)
    assert np.isfinite(tx).all() and not np.array_equal(tx, rx)
    c = Calibration.fit(tx)
    z = np.arange(256,dtype=np.uint8)
    result = replay_bytes(encode_bytes(z, c.build_table(tx)), rx, c)
    assert result.shape == z.shape and result.dtype == np.uint8
    assert np.abs(result.astype(float)-z).mean() < 64

