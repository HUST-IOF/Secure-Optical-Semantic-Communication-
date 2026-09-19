import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def gaussian(x, amp, mu, sigma, offset):
    """
    高斯函数：
    amp    : 顶点幅值
    mu     : 中心位置
    sigma  : 脉冲宽度
    offset : 基线偏移
    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset


def process_one_group(signal_800, num_pulses=7, fit_half_width=20, min_distance=40):
    """
    处理一组 800 点信号，提取 7 个负脉冲并拟合为高斯脉冲。

    参数
    ----
    signal_800 : array-like
        长度为 800 的一组采集信号。
    num_pulses : int
        每组中负脉冲个数，默认 7。
    fit_half_width : int
        每个脉冲左右用于拟合的点数。
    min_distance : int
        find_peaks 中相邻脉冲的最小间隔。

    返回
    ----
    peak_values : ndarray
        7 个高斯脉冲的顶点值，shape = (7,)
        这里输出为正值，表示负脉冲幅度的绝对值。
    peak_positions : ndarray
        7 个脉冲中心位置。
    """
    y = np.asarray(signal_800, dtype=np.float64).flatten()

    if y.size != 800:
        raise ValueError("signal_800 的长度必须是 800")

    # 1. 去基线。负脉冲通常向下，因此用中位数作为基线更稳健
    baseline = np.median(y)
    y0 = y - baseline

    # 2. 将负脉冲取反，变成正峰，方便 find_peaks 和高斯拟合
    y_inv = -y0

    # 3. 找正峰，也就是原始信号中的负脉冲
    peaks, props = find_peaks(
        y_inv,
        distance=min_distance,
        prominence=np.std(y_inv) * 0.5
    )

    if len(peaks) < num_pulses:
        # 如果 prominence 太严格导致找不到 7 个，就放宽条件
        peaks, props = find_peaks(
            y_inv,
            distance=min_distance
        )

    if len(peaks) < num_pulses:
        raise RuntimeError("未找到足够的 7 个负脉冲，请调小 min_distance 或检查信号质量")

    # 4. 选取幅值最大的 7 个峰
    peak_heights = y_inv[peaks]
    selected_idx = np.argsort(peak_heights)[-num_pulses:]
    selected_peaks = peaks[selected_idx]

    # 5. 按位置排序，保证输出顺序对应从左到右 7 个脉冲
    selected_peaks = np.sort(selected_peaks)

    peak_values = []
    fitted_positions = []

    x_all = np.arange(y.size)

    for p in selected_peaks:
        left = max(0, p - fit_half_width)
        right = min(y.size, p + fit_half_width + 1)

        x_fit = x_all[left:right]
        y_fit = y_inv[left:right]

        # 初始参数
        amp0 = np.max(y_fit)
        mu0 = p
        sigma0 = fit_half_width / 3
        offset0 = np.min(y_fit)

        try:
            popt, _ = curve_fit(
                gaussian,
                x_fit,
                y_fit,
                p0=[amp0, mu0, sigma0, offset0],
                bounds=(
                    [0, left, 1e-3, -np.inf],
                    [np.inf, right, fit_half_width * 2, np.inf]
                ),
                maxfev=5000
            )

            amp, mu, sigma, offset = popt

            # 高斯脉冲顶点值 = amp + offset
            # 如果你只想要脉冲幅值，用 amp
            peak_value = amp + offset

        except Exception:
            # 拟合失败时，退化为直接取峰值
            mu = p
            peak_value = y_inv[p]

        peak_values.append(peak_value)
        fitted_positions.append(mu)

    return np.asarray(peak_values), np.asarray(fitted_positions)


def process_full_vector(data, group_len=800, num_pulses=7):
    """
    处理完整 1×204800 向量。

    参数
    ----
    data : array-like
        输入信号，长度应为 204800。
    group_len : int
        每组长度，默认 800。
    num_pulses : int
        每组负脉冲个数，默认 7。

    返回
    ----
    all_peak_values : ndarray
        shape = (256, 7)，每一行对应一个频率组的 7 个高斯脉冲顶点值。
    all_peak_positions : ndarray
        shape = (256, 7)，每一行对应 7 个脉冲中心位置。
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if data.size % group_len != 0:
        raise ValueError("输入长度必须能被 group_len 整除")

    num_groups = data.size // group_len

    all_peak_values = []
    all_peak_positions = []

    for i in range(num_groups):
        seg = data[i * group_len:(i + 1) * group_len]

        peaks, positions = process_one_group(
            seg,
            num_pulses=num_pulses
        )

        all_peak_values.append(peaks)
        all_peak_positions.append(positions)

    return np.asarray(all_peak_values), np.asarray(all_peak_positions)


def get_one_frequency_7d_vector(data, group_index, group_len=800):
    """
    从完整 1×204800 向量中取某一个频率组，输出 7 维向量。

    参数
    ----
    data : array-like
        输入信号，长度 204800。
    group_index : int
        第几个频率组，从 0 开始。
    group_len : int
        每组长度，默认 800。

    返回
    ----
    peak_values : ndarray
        shape = (7,)
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    start = group_index * group_len
    end = start + group_len

    if start < 0 or end > data.size:
        raise ValueError("group_index 超出范围")

    seg = data[start:end]

    peak_values, _ = process_one_group(seg)

    return peak_values