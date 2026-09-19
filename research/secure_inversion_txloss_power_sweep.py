import cv2 as cv
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from lossbuilder import LossBuilder
from quantization import QParam, FakeQuantize
from diffusers import AutoencoderTiny
import argparse

# ===================== Secure physical prompt codec =====================
# This block adds a physical-state-dependent prompt channel.
# Correct communication assumption used here:
#   1) Inversion/training at sender only uses Tx matrix T_tx:
#          f = E(x, T_tx), loss = ||D(MMF(T_tx, f)) - x||
#   2) After inversion, the learned sender prompt bytes are materialized through
#      different Rx matrices T_rx to generate standard .prompt files for 40~50 deg.
#   3) Final .prompt interface is unchanged: U/V and scale/zero_point only.

import os
import re
import glob
import shutil
import scipy.io as sio
from scipy.spatial import cKDTree
from ReflectionCreation.f2VEC import process_full_vector


def _ensure_256x7(A):
    A = np.asarray(A, dtype=np.float64)
    if A.shape == (256, 7):
        return A
    if A.shape == (7, 256):
        return A.T
    raise ValueError("features must be 256x7 or 7x256, got {}".format(A.shape))


def _load_features_from_mat(mat_path):
    sig = sio.loadmat(mat_path)["Data"]
    features_256x7, _ = process_full_vector(sig)
    return _ensure_256x7(features_256x7)


def _build_pca2_params(features_256x7, eps=1e-12):
    A = _ensure_256x7(features_256x7)

    mean_A = np.mean(A, axis=0, keepdims=True)
    A_center = A - mean_A

    _, S, Vt = np.linalg.svd(A_center, full_matrices=False)
    V2 = Vt[:2, :].T
    B = A_center @ V2

    b_min = np.min(B, axis=0, keepdims=True)
    b_max = np.max(B, axis=0, keepdims=True)
    B_norm = 2.0 * (B - b_min) / (b_max - b_min + eps) - 1.0

    return B_norm, {
        "mean_A": mean_A,
        "V2": V2,
        "b_min": b_min,
        "b_max": b_max,
        "singular_values": S,
        "eps": eps,
    }


def _features_to_pca2_norm(features_256x7, pca_params):
    A = _ensure_256x7(features_256x7)
    eps = pca_params.get("eps", 1e-12)

    A_center = A - pca_params["mean_A"]
    B = A_center @ pca_params["V2"]
    B_norm = 2.0 * (B - pca_params["b_min"]) / (
        pca_params["b_max"] - pca_params["b_min"] + eps
    ) - 1.0
    return B_norm


def _all_2d_targets():
    values = np.arange(256, dtype=np.uint16)
    z0, z1 = np.meshgrid(values, values, indexing="ij")
    z_pairs = np.stack([z0.reshape(-1), z1.reshape(-1)], axis=1).astype(np.uint8)
    target = 2.0 * z_pairs.astype(np.float64) / 255.0 - 1.0
    return z_pairs, target


def _build_single_norm_diff_clip_table(features_256x7, gain=4.0):
    B_norm, pca_params = _build_pca2_params(features_256x7)
    K = B_norm.shape[0]

    i_idx, j_idx = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    i_idx = i_idx.reshape(-1)
    j_idx = j_idx.reshape(-1)

    pair_indices = np.stack([i_idx, j_idx], axis=1).astype(np.uint16)
    pair_vectors = gain * (B_norm[i_idx, :] - B_norm[j_idx, :])
    pair_vectors = np.clip(pair_vectors, -1.0, 1.0)

    z_pairs, target = _all_2d_targets()
    tree = cKDTree(pair_vectors)

    try:
        dist, nn_idx = tree.query(target, k=1, workers=-1)
    except TypeError:
        dist, nn_idx = tree.query(target, k=1)

    selected = pair_indices[nn_idx]
    matched = pair_vectors[nn_idx]
    loss = (dist ** 2) / 2.0

    z_hat = np.clip(
        np.round((matched + 1.0) * 0.5 * 255.0),
        0,
        255
    ).astype(np.uint8)

    abs_err = np.abs(z_hat.astype(np.int16) - z_pairs.astype(np.int16))

    pca_params["gain"] = gain
    return selected, loss, pca_params, abs_err


def ensure_trans_table(trans_mat_path, table_path, gain=4.0):
    """Build the Tx lookup table once. Reused for all matrices and frames."""
    if os.path.exists(table_path):
        return table_path

    os.makedirs(os.path.dirname(table_path), exist_ok=True)

    features_256x7 = _load_features_from_mat(trans_mat_path)
    selected, loss, pca_params, abs_err = _build_single_norm_diff_clip_table(
        features_256x7,
        gain=gain
    )

    np.savez_compressed(
        table_path,
        selected=selected.astype(np.uint16),
        loss=loss.astype(np.float32),
        mean_A=pca_params["mean_A"],
        V2=pca_params["V2"],
        b_min=pca_params["b_min"],
        b_max=pca_params["b_max"],
        singular_values=pca_params["singular_values"],
        eps=np.array(pca_params["eps"]),
        gain=np.array(pca_params["gain"]),
    )

    print("physical lookup table saved:", table_path)
    print("table mean loss:", float(np.mean(loss)))
    print("table max loss:", float(np.max(loss)))
    print("table mean MAE8:", float(np.mean(abs_err)))
    print("table max err8:", int(np.max(abs_err)))
    return table_path


def load_trans_table(table_path):
    data = np.load(table_path, allow_pickle=True)
    selected = data["selected"].astype(np.int64)

    pca_params = {
        "mean_A": data["mean_A"],
        "V2": data["V2"],
        "b_min": data["b_min"],
        "b_max": data["b_max"],
        "singular_values": data["singular_values"],
        "eps": float(data["eps"]),
        "gain": float(data["gain"]) if "gain" in data.files else 4.0,
    }
    return selected, pca_params


class PhysicalPromptCodec:
    """
    Optimized continuous converter.

    It reads the Tx lookup table and one physical matrix once, then converts
    any uint8 matrix repeatedly:
        sender uint8 -> frequency pair selected by Tx table -> recovered uint8

    For inversion, pass receive_mat_path=trans_mat_path so that the loss uses
    only the sender-known Tx matrix. For final receiver outputs, instantiate
    with receive_mat_path set to the desired Rx matrix.

    tensor_through_channel() uses STE:
        forward: physical recovered tensor
        backward: identity gradient to Q_U/Q_V
    """

    def __init__(self, trans_mat_path, receive_mat_path, table_path, gain=4.0):
        self.trans_mat_path = trans_mat_path
        self.receive_mat_path = receive_mat_path
        self.table_path = table_path
        self.gain = float(gain)

        ensure_trans_table(trans_mat_path, table_path, gain=gain)
        self.selected, self.pca_params = load_trans_table(table_path)
        self.gain = float(self.pca_params.get("gain", gain))

        rx_features = _load_features_from_mat(receive_mat_path)
        self.receive_pca_256x2 = _features_to_pca2_norm(rx_features, self.pca_params)

    def transform_uint8(self, m):
        m = np.asarray(m, dtype=np.uint8)
        original_shape = m.shape

        m_flat = m.reshape(-1)
        original_len = m_flat.size

        if original_len % 2 != 0:
            m_flat = np.concatenate([m_flat, np.array([0], dtype=np.uint8)])
            pad_len = 1
        else:
            pad_len = 0

        m_pairs = m_flat.reshape(-1, 2).astype(np.uint8)
        pair_index = (
            m_pairs[:, 0].astype(np.int64) * 256
            + m_pairs[:, 1].astype(np.int64)
        )

        freq_pairs = self.selected[pair_index]
        f1 = freq_pairs[:, 0].astype(np.int64)
        f2 = freq_pairs[:, 1].astype(np.int64)

        q_hat = self.gain * (
            self.receive_pca_256x2[f1, :] - self.receive_pca_256x2[f2, :]
        )
        q_hat = np.clip(q_hat, -1.0, 1.0)

        recovered_pairs = np.clip(
            np.round((q_hat + 1.0) * 0.5 * 255.0),
            0,
            255
        ).astype(np.uint8)

        recovered_flat = recovered_pairs.reshape(-1)
        if pad_len > 0:
            recovered_flat = recovered_flat[:-pad_len]

        return recovered_flat.reshape(original_shape).astype(np.uint8)

    @staticmethod
    def _dequantize_uint8(byte_np, qparam, device):
        byte_t = torch.from_numpy(byte_np.astype(np.float32)).to(device=device)

        scale = qparam.scale
        zero_point = qparam.zero_point

        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, dtype=torch.float32, device=device)
        else:
            scale = scale.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(zero_point):
            zero_point = torch.tensor(zero_point, dtype=torch.float32, device=device)
        else:
            zero_point = zero_point.to(device=device, dtype=torch.float32)

        return (byte_t - zero_point) * scale

    def tensor_through_channel(self, q_tensor, qparam):
        """
        q_tensor is fake-quantized float tensor.

        Returns:
            out: recovered float tensor with STE gradient.
            rec_byte: recovered uint8 byte tensor after the physical matrix.
            sender_byte: original sender uint8 byte tensor before the physical matrix.

        During inversion, the forward loss should use out. The saved source
        prompt should use sender_byte, because receiver-temperature prompt files
        must be materialized later from the actually transmitted frequency code.
        """
        sender_byte_np = qparam.quantize_tensor(q_tensor).detach().cpu().numpy().astype(np.uint8)
        rec_byte_np = self.transform_uint8(sender_byte_np)
        rec_float = self._dequantize_uint8(rec_byte_np, qparam, q_tensor.device).to(dtype=q_tensor.dtype)

        # STE: forward = rec_float, backward = q_tensor
        out = q_tensor + (rec_float - q_tensor).detach()
        rec_byte = torch.from_numpy(rec_byte_np).to(device=q_tensor.device, dtype=torch.uint8)
        sender_byte = torch.from_numpy(sender_byte_np).to(device=q_tensor.device, dtype=torch.uint8)
        return out, rec_byte, sender_byte


def collect_data_files(data_root):
    pattern = os.path.join(data_root, "Data20260508_20km_*MHz_*deg_*.mat")
    files = sorted(glob.glob(pattern))
    regex = re.compile(
        r"Data20260508_20km_(?P<freq>[\d.]+MHz)_(?P<temp>\d+)deg_(?P<rep>\d+)\.mat$"
    )

    file_dict = {}
    for fp in files:
        name = os.path.basename(fp)
        m = regex.match(name)
        if m is None:
            continue
        freq = m.group("freq")
        temp = int(m.group("temp"))
        rep = int(m.group("rep"))
        file_dict.setdefault(freq, {}).setdefault(temp, {})[rep] = fp
    return file_dict


def freq_sort_key(freq):
    return float(freq.replace("MHz", ""))


def prepare_frame_folder(src_frame_path, dst_frame_path, overwrite=True):
    """
    Copy images and metadata to a condition-specific folder.
    The output interface remains the same as original inversion:
        dst_frame_path/results/rank{rank}_interval{interval}/frame_xxxxx.prompt
    """
    src_frame_path = os.path.abspath(src_frame_path)
    dst_frame_path = os.path.abspath(dst_frame_path)

    if src_frame_path == dst_frame_path:
        return dst_frame_path

    if os.path.exists(dst_frame_path):
        if overwrite:
            shutil.rmtree(dst_frame_path)
        else:
            return dst_frame_path

    def _ignore(dirpath, names):
        return {"results"} if "results" in names else set()

    shutil.copytree(src_frame_path, dst_frame_path, ignore=_ignore)
    return dst_frame_path


def _prompt_dir(frame_path, rank, interval):
    return os.path.join(frame_path, 'results/rank{}_interval{}'.format(rank, interval))


def materialize_receiver_prompts(
        src_frame_path,
        dst_frame_path,
        rank,
        interval,
        codec,
        overwrite_prompts=True,
        save_device="cuda:0"
):
    """
    Convert sender-side prompt bytes into receiver-side effective prompt bytes.

    Input source prompts keep the original inversion interface:
        U, V, U_scale, U_zero_point, V_scale, V_zero_point

    Output receiver prompts keep exactly the same interface. No extra fields are added.
    """
    src_prompt_dir = _prompt_dir(src_frame_path, rank, interval)
    dst_prompt_dir = _prompt_dir(dst_frame_path, rank, interval)
    os.makedirs(dst_prompt_dir, exist_ok=True)

    # copy init.pth if present; generation.py expects the same original interface
    src_init = os.path.join(src_prompt_dir, 'init.pth')
    if os.path.exists(src_init):
        shutil.copy2(src_init, os.path.join(dst_prompt_dir, 'init.pth'))

    prompt_files = sorted(glob.glob(os.path.join(src_prompt_dir, 'frame_*.prompt')))
    if len(prompt_files) == 0:
        raise FileNotFoundError('No frame_*.prompt found in {}'.format(src_prompt_dir))

    mae_u_list = []
    mae_v_list = []

    for src_prompt in prompt_files:
        dst_prompt = os.path.join(dst_prompt_dir, os.path.basename(src_prompt))
        if os.path.exists(dst_prompt) and not overwrite_prompts:
            continue

        prompt = torch.load(src_prompt, map_location='cpu', weights_only=False)

        U_sender = prompt['U'].detach().cpu().to(torch.uint8).numpy()
        V_sender = prompt['V'].detach().cpu().to(torch.uint8).numpy()

        U_rx = codec.transform_uint8(U_sender)
        V_rx = codec.transform_uint8(V_sender)

        mae_u_list.append(float(np.mean(np.abs(U_sender.astype(np.float64) - U_rx.astype(np.float64)))))
        mae_v_list.append(float(np.mean(np.abs(V_sender.astype(np.float64) - V_rx.astype(np.float64)))))

        def _to_save_device(v):
            if torch.is_tensor(v):
                return v.detach().to(device=save_device)
            return v

        out_prompt = {
            'U': torch.from_numpy(U_rx).to(device=save_device, dtype=torch.uint8),
            'V': torch.from_numpy(V_rx).to(device=save_device, dtype=torch.uint8),
            'U_scale': _to_save_device(prompt['U_scale']),
            'U_zero_point': _to_save_device(prompt['U_zero_point']),
            'V_scale': _to_save_device(prompt['V_scale']),
            'V_zero_point': _to_save_device(prompt['V_zero_point']),
        }
        torch.save(out_prompt, dst_prompt)

    # write a small conversion log; generation interface is unaffected
    with open(os.path.join(dst_prompt_dir, 'physical_materialize_log.txt'), 'w') as f:
        f.write('num_prompts: {}\n'.format(len(prompt_files)))
        f.write('mean_U_MAE8: {}\n'.format(float(np.mean(mae_u_list)) if mae_u_list else None))
        f.write('mean_V_MAE8: {}\n'.format(float(np.mean(mae_v_list)) if mae_v_list else None))
        f.write('save_device: {}\n'.format(save_device))

    print('materialized receiver prompts:', dst_prompt_dir)
    print('mean U MAE8:', float(np.mean(mae_u_list)) if mae_u_list else None)
    print('mean V MAE8:', float(np.mean(mae_v_list)) if mae_v_list else None)

    return dst_prompt_dir


VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def inversion(
        model,
        sampler,
        decoder,
        rank,
        interval,
        frame_path,
        max_id,
        H=512,
        W=512,
        seed=0,
        filter=None,
        physical_codec=None
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        uc = None
        rand_noise = seeded_randn(shape, seed)
        sigma = torch.Tensor([0.05]).float().cuda()

        # Set the loss functions.
        # Reconstruction loss.
        mse_loss = torch.nn.MSELoss()
        # Perceptual loss.
        builder = LossBuilder('cuda')
        content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                          ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                          ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                          ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                          ('conv_16', 1)]
        vgg_model, lpips_nodes = builder.get_style_and_content_loss(dict(content_layers))

        # Inversion.
        for f_id in range(0, max_id, interval):
            # Initialize the low-rank factor U.
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            # Fake Quantizer for U
            Quant_Param_U = QParam(num_bits=8)
            # Initialize the low-rank factor V.
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            # Fake Quantizer for V
            Quant_Param_V = QParam(num_bits=8)

            # Initialize the learning rate and the optimizer.
            lr = 0.1
            optimizer = torch.optim.Adam([U, V], lr=lr)

            # for learning rate scheduler and logging
            min_loss = 1e9
            latest_min_loss = 1e9

            # logs and results path
            prompt_path = os.path.join(frame_path, 'results/rank{}_interval{}/'.format(rank, interval))
            log_path = os.path.join(prompt_path,'{:05d}'.format(f_id))
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_output = open(os.path.join(log_path,'log.txt'), 'a')

            if f_id > 0:
                ckpt_prev = torch.load(os.path.join(prompt_path,'{:05d}/ckpt.pth'.format(f_id - interval)),weights_only=False)
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
                prev_frame = ckpt_prev["z"]
                # Subsequent frames require fewer iterations.
                # Reduce total_iterations to speed up inversion, but this may lower quality.
                total_iterations = 1500
                lr_schedule_cnt = 20
                step_list_base = [_ for _ in range(1, interval + 1)]
            else:
                # Initialization of the first frame.
                prev_frame = model.encode_first_stage(load_img(os.path.join(frame_path,'00000.png')))
                torch.save(prev_frame, os.path.join(prompt_path, 'init.pth'))
                # The first frame requires more iterations.
                total_iterations = 10000
                lr_schedule_cnt = 300
                step_list_base = [0]
            # add random noise to the previous frame
            randn = (prev_frame * sigma + rand_noise * (1 - sigma)).detach()
            step_list = step_list_base

            for iter in range(total_iterations):
                loss_list = {}
                for step in step_list:
                    # Fake Quantification
                    Quant_Param_U.update(U)
                    Q_U = FakeQuantize.apply(U, Quant_Param_U)
                    Quant_Param_V.update(V)
                    Q_V = FakeQuantize.apply(V, Quant_Param_V)

                    # Physical prompt channel during inversion:
                    #   sender quantized prompt bytes -> frequency code -> MMF(T_tx) recovered bytes.
                    # IMPORTANT: physical_codec must use receive_mat_path=trans_mat_path here,
                    # because the sender does not know the receiver matrix during optimization.
                    # STE keeps gradients flowing to U and V.
                    if physical_codec is not None:
                        U_effect, U_Byte_effect, U_Byte_sender = physical_codec.tensor_through_channel(Q_U, Quant_Param_U)
                        V_effect, V_Byte_effect, V_Byte_sender = physical_codec.tensor_through_channel(Q_V, Quant_Param_V)
                    else:
                        U_effect, U_Byte_effect = Q_U, None
                        V_effect, V_Byte_effect = Q_V, None
                        U_Byte_sender = None
                        V_Byte_sender = None

                    if f_id > 0:
                        # perform linear interpolation on keyframe prompts
                        # approximating the intermediate prompts.
                        factor = 1 / interval
                        u = (1 - step * factor) * U_prev + (step * factor) * U_effect
                        v = (1 - step * factor) * V_prev + (step * factor) * V_effect
                        # prompt composition
                        c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = f_id - interval + step
                    else:
                        # for the first frame
                        c = (U_effect @ V_effect / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = 0
                    c = {'crossattn': c}

                    # generating a frame
                    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                    samples_x = decoder(samples_z)

                    # Calculating loss
                    gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(cur_id)))
                    gt.requires_grad = True
                    # Perceptual loss.
                    vgg_model(torch.cat([gt, samples_x], dim=0))
                    lpips_loss = 0
                    for node in lpips_nodes:
                        lpips_loss += node.loss
                    lpips_loss = lpips_loss / (len(lpips_nodes) + 1e-9)
                    # Combine the perceptual loss and reconstruction loss.
                    loss = 0.2 * lpips_loss + 0.8 * mse_loss(samples_x, gt)

                    # regularization
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu

                    # logging
                    print('iter: {}, cur_id: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}'.format(iter, cur_id, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.write('iter: {}, cur_id: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}\n'.format(iter, cur_id, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.flush()

                    # saving the generated frames
                    if iter % 100 == 0:
                        img = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                        if filter is not None:
                            img = filter(img)
                        img = (
                            (255 * img)
                                .to(dtype=torch.uint8)
                                .permute(0, 2, 3, 1)
                                .detach()
                                .cpu()
                                .numpy()
                        )
                        img = img[0][:, :, ::-1]
                        cv2.imwrite(os.path.join(log_path,'iter_{:05d}_id_{:05d}.png'.format(iter, cur_id)), img)

                    # Optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.model.zero_grad()

                    # for learning rate scheduler and logging
                    if cur_id not in loss_list.keys():
                        loss_list[cur_id] = loss.item()
                mean_loss = np.mean(list(loss_list.values()))
                print('iter: {}, mean loss: {}'.format(iter, mean_loss))
                log_output.write('iter: {}, mean loss: {}\n'.format(iter, mean_loss))
                log_output.flush()
                if mean_loss < min_loss:
                    # saving the ckpt
                    min_loss = mean_loss
                    ckpt = {
                        'U': U_effect,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'U_bits': Quant_Param_U.num_bits,
                        'V': V_effect,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                        'V_bits': Quant_Param_V.num_bits,
                        'z': samples_z,
                        'randn': randn,
                        'iter': iter,
                        'loss': mean_loss,
                    }
                    torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))
                    # saving the source prompt
                    # Save the sender-side bytes, not the Tx-recovered bytes.
                    # These bytes represent the transmitted frequency code and will be
                    # materialized into receiver-side prompts after inversion.
                    if U_Byte_sender is None:
                        U_Byte = Quant_Param_U.quantize_tensor(Q_U).byte()
                    else:
                        U_Byte = U_Byte_sender.byte()
                    if V_Byte_sender is None:
                        V_Byte = Quant_Param_V.quantize_tensor(Q_V).byte()
                    else:
                        V_Byte = V_Byte_sender.byte()
                    prompt = {
                        'U': U_Byte,
                        'V': V_Byte,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                    }
                    torch.save(prompt, os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(f_id)))
                if f_id > 0:
                    # Dynamic training.
                    # Allocating more training to the frames with the worst performance.
                    worst_step = max(loss_list, key=loss_list.get) - f_id + interval
                    step_list = step_list_base + [worst_step] * 2
                    step_list = sorted(step_list)
                # Learning rate scheduler
                lr_schedule_cnt = lr_schedule_cnt - 1
                if lr_schedule_cnt == 0:
                    if min_loss == latest_min_loss:
                        # Reduce the learning rate by half.
                        lr = max(lr * 0.5, 0.001)
                        optimizer = torch.optim.Adam([U, V], lr=lr)
                        print('reduce lr to: {}'.format(optimizer.param_groups[0]['lr']))
                        log_output.write('reduce lr to: {}\n'.format(optimizer.param_groups[0]['lr']))
                        log_output.flush()
                    latest_min_loss = min_loss
                    lr_schedule_cnt = 20 if f_id > 0 else 300
            log_output.close()


def build_model_sampler_decoder():
    print('0')
    version_dict = VERSION2SPECS['SD-Turbo']
    print('1')
    state = init_st(version_dict, load_filter=True)
    print('2')
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    print('3')
    load_model(model)

    taesd = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd",
        torch_dtype=torch.float32
    ).cuda()

    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    seed_ = 88
    sampler.noise_sampler = SeededNoise(seed=seed_)
    return state, model, sampler, taesd, seed_


def run_secure_sweep(args, model, sampler, taesd, state, seed_):
    """
    Correct version:
      - For each frequency, inversion is performed once using only Tx matrix T_tx.
      - The learned source prompts are sender-side bytes.
      - After inversion, source prompts are materialized through Rx matrices for 40~50 deg.
      - Final receiver folders keep the same prompt interface as original inversion.py.
    """
    file_dict = collect_data_files(args.data_root)

    if args.freq is None or args.freq.lower() == "all":
        freqs = sorted(file_dict.keys(), key=freq_sort_key)
    else:
        # 支持:
        # --freq 40MHz
        # --freq 10MHz,20MHz,30MHz
        # --freq 10-100MHz
        freq_arg = args.freq.strip()

        if "," in freq_arg:
            freqs = [f.strip() for f in freq_arg.split(",") if f.strip()]

        elif "-" in freq_arg and freq_arg.endswith("MHz"):
            m = re.match(r"([0-9.]+)-([0-9.]+)MHz$", freq_arg)
            if m is None:
                raise ValueError(f"Invalid freq range: {freq_arg}")
            start_f = float(m.group(1))
            end_f = float(m.group(2))

            freqs = []
            for f in sorted(file_dict.keys(), key=freq_sort_key):
                fv = float(f.replace("MHz", ""))
                if start_f <= fv <= end_f:
                    freqs.append(f)
        else:
            freqs = [freq_arg]

    temps = [int(t) for t in args.temps.split(",")]

    table_root = os.path.join(args.output_root, "_tables")
    source_root = os.path.join(args.output_root, "_tx_inversion_source")
    os.makedirs(table_root, exist_ok=True)
    os.makedirs(source_root, exist_ok=True)

    for freq in freqs:
        if freq not in file_dict:
            print("skip {}, no data".format(freq))
            continue

        if args.trans_temp not in file_dict[freq]:
            print("skip {}, no {}deg Tx data".format(freq, args.trans_temp))
            continue

        if args.trans_rep not in file_dict[freq][args.trans_temp]:
            print("skip {}, no {}deg_R{} Tx data".format(freq, args.trans_temp, args.trans_rep))
            continue

        trans_mat_path = file_dict[freq][args.trans_temp][args.trans_rep]
        freq_tag = freq.replace(".", "p")

        table_path = os.path.join(
            table_root,
            "trans_table_{}_Tx{}degR{}_gain{}.npz".format(
                freq_tag, args.trans_temp, args.trans_rep, args.gain
            )
        )

        source_frame_path = os.path.join(
            source_root,
            "{}_Tx{}degR{}_sender_loss_gain{}".format(
                freq_tag, args.trans_temp, args.trans_rep, args.gain
            )
        )

        prepare_frame_folder(args.frame_path, source_frame_path, overwrite=args.overwrite)

        print("\n==============================")
        print("freq:", freq)
        print("Tx inversion matrix:", trans_mat_path)
        print("source frame_path:", source_frame_path)
        print("table:", table_path)
        print("NOTE: inversion loss uses Tx matrix only.")

        # Inversion uses Tx as both table source and physical forward matrix.
        # This matches the assumption that the sender does not know T_rx.
        tx_codec = PhysicalPromptCodec(
            trans_mat_path=trans_mat_path,
            receive_mat_path=trans_mat_path,
            table_path=table_path,
            gain=args.gain
        )

        inversion(
            model,
            sampler,
            decoder=taesd.decoder,
            rank=args.rank,
            interval=args.interval,
            frame_path=source_frame_path,
            max_id=args.max_id,
            H=512,
            W=512,
            seed=seed_,
            filter=state.get("filter"),
            physical_codec=tx_codec
        )

        # After Tx-only inversion, materialize receiver-side prompts for each Rx temperature.
        for temp in temps:
            if temp not in file_dict[freq]:
                print("skip {} {}deg, no Rx data".format(freq, temp))
                continue

            available_reps = sorted(file_dict[freq][temp].keys())

            if temp == args.trans_temp:
                valid_reps = [r for r in available_reps if r != args.trans_rep]
                if len(valid_reps) == 0:
                    print("skip {} {}deg, no different Rx rep".format(freq, temp))
                    continue
                recv_rep = args.recv_rep if args.recv_rep in valid_reps else valid_reps[0]
            else:
                recv_rep = args.recv_rep if args.recv_rep in available_reps else available_reps[0]

            receive_mat_path = file_dict[freq][temp][recv_rep]

            dst_frame_path = os.path.join(
                args.output_root,
                "{}_Tx{}degR{}_Rx{}degR{}_gain{}".format(
                    freq_tag, args.trans_temp, args.trans_rep, temp, recv_rep, args.gain
                )
            )

            prepare_frame_folder(args.frame_path, dst_frame_path, overwrite=args.overwrite)

            print("\n--- materialize Rx prompts ---")
            print("freq:", freq)
            print("Rx:", receive_mat_path)
            print("dst frame_path:", dst_frame_path)

            rx_codec = PhysicalPromptCodec(
                trans_mat_path=trans_mat_path,
                receive_mat_path=receive_mat_path,
                table_path=table_path,
                gain=args.gain
            )

            materialize_receiver_prompts(
                src_frame_path=source_frame_path,
                dst_frame_path=dst_frame_path,
                rank=args.rank,
                interval=args.interval,
                codec=rx_codec,
                overwrite_prompts=True
            )


# ===================== Power / intensity sweep for 20260509 data =====================

def collect_power_data_files(data_root):
    """
    Scan files like:
        Data20260509_20km_60MHz_50deg_1_1dBm.mat
        Data20260509_20km_60MHz_40deg_2_3dBm.mat

    Return:
        file_dict[freq][power_tag][temp][rep] = filepath
    where power_tag is like '1dBm', '2dBm', '3dBm'.
    """
    pattern = os.path.join(data_root, "Data*_20km_*MHz_*deg_*_*dBm.mat")
    files = sorted(glob.glob(pattern))
    regex = re.compile(
        r"Data\d+_20km_(?P<freq>[\d.]+MHz)_(?P<temp>\d+)deg_(?P<rep>\d+)_(?P<power>-?[\d.]+)dBm\.mat$"
    )

    file_dict = {}
    for fp in files:
        name = os.path.basename(fp)
        m = regex.match(name)
        if m is None:
            continue
        freq = m.group("freq")
        temp = int(m.group("temp"))
        rep = int(m.group("rep"))
        power_tag = "{}dBm".format(m.group("power"))
        file_dict.setdefault(freq, {}).setdefault(power_tag, {}).setdefault(temp, {})[rep] = fp
    return file_dict


def power_sort_key(power_tag):
    return float(power_tag.replace("dBm", ""))


def parse_power_selection(power_arg, available_power_tags):
    """
    Support:
        --power all
        --power 1dBm
        --power 1,2,3
        --power 1dBm,2dBm,3dBm
        --power 1-3dBm
    """
    if power_arg is None or power_arg.lower() == "all":
        return sorted(available_power_tags, key=power_sort_key)

    s = power_arg.strip()

    if "," in s:
        out = []
        for item in s.split(","):
            item = item.strip()
            if not item:
                continue
            if not item.endswith("dBm"):
                item = item + "dBm"
            out.append(item)
        return out

    if "-" in s and s.endswith("dBm"):
        m = re.match(r"(-?[0-9.]+)-(-?[0-9.]+)dBm$", s)
        if m is None:
            raise ValueError("Invalid power range: {}".format(s))
        p0 = float(m.group(1))
        p1 = float(m.group(2))
        out = []
        for p in sorted(available_power_tags, key=power_sort_key):
            pv = power_sort_key(p)
            if p0 <= pv <= p1:
                out.append(p)
        return out

    if not s.endswith("dBm"):
        s = s + "dBm"
    return [s]


def run_power_sweep(args, model, sampler, taesd, state, seed_):
    """
    Same logic as secure tx-loss inversion, but adds an intensity / power dimension.

    For each selected power:
      1) Tx inversion from images is run from scratch.
         Tx matrix: freq, trans_temp=50deg, trans_rep=1, selected power.
         Loss uses Tx matrix only.
      2) The sender-side prompts are materialized into Rx prompts for 40~50deg
         using the same frequency and same power, with recv_rep selected as before.

    No previous prompt files are used.
    """
    if "cuda" in args.save_device and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --save_device uses CUDA.")

    file_dict = collect_power_data_files(args.data_root)

    if args.freq not in file_dict:
        raise FileNotFoundError("No data found for freq {} under {}".format(args.freq, args.data_root))

    available_powers = sorted(file_dict[args.freq].keys(), key=power_sort_key)
    powers = parse_power_selection(args.power, available_powers)
    temps = [int(t) for t in args.temps.split(",")]

    table_root = os.path.join(args.output_root, "_tables")
    source_root = os.path.join(args.output_root, "_tx_inversion_source")
    os.makedirs(table_root, exist_ok=True)
    os.makedirs(source_root, exist_ok=True)

    print("available powers:", available_powers)
    print("selected powers:", powers)
    print("selected temps:", temps)

    for power_tag in powers:
        if power_tag not in file_dict[args.freq]:
            print("skip power {}, no data".format(power_tag))
            continue

        pdict = file_dict[args.freq][power_tag]

        if args.trans_temp not in pdict:
            print("skip {} {}, no {}deg Tx data".format(args.freq, power_tag, args.trans_temp))
            continue

        if args.trans_rep not in pdict[args.trans_temp]:
            print("skip {} {}, no {}deg_R{} Tx data".format(
                args.freq, power_tag, args.trans_temp, args.trans_rep
            ))
            continue

        trans_mat_path = pdict[args.trans_temp][args.trans_rep]
        freq_tag = args.freq.replace(".", "p")
        power_file_tag = power_tag.replace(".", "p")

        table_path = os.path.join(
            table_root,
            "trans_table_{}_{}_Tx{}degR{}_gain{}.npz".format(
                freq_tag, power_file_tag, args.trans_temp, args.trans_rep, args.gain
            )
        )

        source_frame_path = os.path.join(
            source_root,
            "{}_{}_Tx{}degR{}_sender_loss_gain{}".format(
                freq_tag, power_file_tag, args.trans_temp, args.trans_rep, args.gain
            )
        )

        prepare_frame_folder(args.frame_path, source_frame_path, overwrite=args.overwrite)

        print("\n==============================")
        print("freq:", args.freq)
        print("power:", power_tag)
        print("Tx inversion matrix:", trans_mat_path)
        print("source frame_path:", source_frame_path)
        print("table:", table_path)
        print("NOTE: inversion is run from images and loss uses Tx matrix only.")

        tx_codec = PhysicalPromptCodec(
            trans_mat_path=trans_mat_path,
            receive_mat_path=trans_mat_path,
            table_path=table_path,
            gain=args.gain
        )

        inversion(
            model,
            sampler,
            decoder=taesd.decoder,
            rank=args.rank,
            interval=args.interval,
            frame_path=source_frame_path,
            max_id=args.max_id,
            H=512,
            W=512,
            seed=seed_,
            filter=state.get("filter"),
            physical_codec=tx_codec
        )

        for temp in temps:
            if temp not in pdict:
                print("skip {} {} {}, no Rx data".format(args.freq, power_tag, temp))
                continue

            available_reps = sorted(pdict[temp].keys())

            if temp == args.trans_temp:
                valid_reps = [r for r in available_reps if r != args.trans_rep]
                if len(valid_reps) == 0:
                    print("skip {} {} {}deg, no different Rx rep".format(args.freq, power_tag, temp))
                    continue
                recv_rep = args.recv_rep if args.recv_rep in valid_reps else valid_reps[0]
            else:
                recv_rep = args.recv_rep if args.recv_rep in available_reps else available_reps[0]

            receive_mat_path = pdict[temp][recv_rep]

            dst_frame_path = os.path.join(
                args.output_root,
                "{}_{}_Tx{}degR{}_Rx{}degR{}_gain{}".format(
                    freq_tag, power_file_tag, args.trans_temp, args.trans_rep,
                    temp, recv_rep, args.gain
                )
            )

            prepare_frame_folder(args.frame_path, dst_frame_path, overwrite=args.overwrite)

            print("\n--- materialize Rx prompts ---")
            print("freq:", args.freq)
            print("power:", power_tag)
            print("Rx:", receive_mat_path)
            print("dst frame_path:", dst_frame_path)

            rx_codec = PhysicalPromptCodec(
                trans_mat_path=trans_mat_path,
                receive_mat_path=receive_mat_path,
                table_path=table_path,
                gain=args.gain
            )

            materialize_receiver_prompts(
                src_frame_path=source_frame_path,
                dst_frame_path=dst_frame_path,
                rank=args.rank,
                interval=args.interval,
                codec=rx_codec,
                overwrite_prompts=True,
                save_device=args.save_device
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Original inversion arguments
    parser.add_argument('-frame_path', type=str, default="data/sky")
    parser.add_argument('-max_id', type=int, default=140)
    parser.add_argument('-rank', type=int, default=8)
    parser.add_argument('-interval', type=int, default=10)

    # Secure physical channel arguments
    parser.add_argument('--secure_sweep', action='store_true',
                        help='Run 40~50 deg physical secure inversion sweep.')
    parser.add_argument('--data_root', type=str, default="../data/measured")
    parser.add_argument('--output_root', type=str, default="../runs/secure_inversion_sweep")
    parser.add_argument('--freq', type=str, default="40MHz",
                        help='Frequency selection. Examples: 40MHz ; 10MHz,20MHz,30MHz ; 10-100MHz ; all')
    parser.add_argument('--temps', type=str, default="40,42,44,46,48,50")
    parser.add_argument('--trans_temp', type=int, default=50)
    parser.add_argument('--trans_rep', type=int, default=1)
    parser.add_argument('--recv_rep', type=int, default=2)
    parser.add_argument('--gain', type=float, default=4.0)
    parser.add_argument('--overwrite', action='store_true')

    # Power/intensity sweep arguments for 20260509 data.
    parser.add_argument('--power_sweep', action='store_true',
                        help='Run same-frequency, different-intensity secure inversion sweep. Uses Data*_..._*dBm.mat files.')
    parser.add_argument('--power', type=str, default='all',
                        help='Power selection, e.g. 1dBm ; 1,2,3 ; 1-3dBm ; all')
    parser.add_argument('--save_device', type=str, default='cuda:0',
                        help='Device for saved receiver .prompt tensors. Use cuda:0 to match generation.py.')

    # Optional single secure condition.
    parser.add_argument('--trans_mat_path', type=str, default=None)
    parser.add_argument('--receive_mat_path', type=str, default=None)
    parser.add_argument('--table_path', type=str, default=None)

    args = parser.parse_args()

    state, model, sampler, taesd, seed_ = build_model_sampler_decoder()

    if args.power_sweep:
        run_power_sweep(args, model, sampler, taesd, state, seed_)
    elif args.secure_sweep:
        run_secure_sweep(args, model, sampler, taesd, state, seed_)
    else:
        physical_codec = None
        need_materialize_rx = False
        table_path = args.table_path

        if args.trans_mat_path is not None:
            if table_path is None:
                table_path = os.path.join(
                    args.frame_path,
                    "results",
                    "rank{}_interval{}".format(args.rank, args.interval),
                    "trans_table_gain{}.npz".format(args.gain)
                )

            # Correct assumption for single-condition mode too:
            # inversion loss uses Tx matrix only.
            physical_codec = PhysicalPromptCodec(
                trans_mat_path=args.trans_mat_path,
                receive_mat_path=args.trans_mat_path,
                table_path=table_path,
                gain=args.gain
            )

            if args.receive_mat_path is not None:
                need_materialize_rx = True

        inversion(
            model,
            sampler,
            decoder=taesd.decoder,
            rank=args.rank,
            interval=args.interval,
            frame_path=args.frame_path,
            max_id=args.max_id,
            H=512,
            W=512,
            seed=seed_,
            filter=state.get("filter"),
            physical_codec=physical_codec
        )

        # In single-condition mode, if receive_mat_path is provided, overwrite
        # frame_path prompts with receiver-side effective prompts after Tx-only inversion.
        # The saved prompt interface remains unchanged.
        if need_materialize_rx:
            rx_codec = PhysicalPromptCodec(
                trans_mat_path=args.trans_mat_path,
                receive_mat_path=args.receive_mat_path,
                table_path=table_path,
                gain=args.gain
            )
            materialize_receiver_prompts(
                src_frame_path=args.frame_path,
                dst_frame_path=args.frame_path,
                rank=args.rank,
                interval=args.interval,
                codec=rx_codec,
                overwrite_prompts=True
            )
