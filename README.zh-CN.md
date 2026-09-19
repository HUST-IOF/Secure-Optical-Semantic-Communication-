# 安全光语义通信：代码与实测样例

本项目对应稿件 **Secure optical semantic communication through reconfigurable fiber responses**。

## 最快运行方式

在项目根目录执行：

```bash
python -m pip install -e ".[test]"
python examples/codec_demo.py
python examples/codec_demo.py --measured
python -m pytest -q
```

第二个示例读取真实 70 MHz 波形，完成七峰提取、离线 PCA、频率对编码和响应差分恢复。输入是固定随机种子的 8,808 字节测试向量；无需下载图像生成模型。该示例属于实测码本重放，不是现场光链路传输，也不是神经攻击安全验证。

## GPU

先根据 [PyTorch 官方页面](https://pytorch.org/get-started/locally/)安装与你的显卡驱动匹配的 CUDA 版本，再执行：

```bash
python -m pip install -e ".[test,gpu]"
python examples/torch_receiver_demo.py --device cuda
python examples/attack_smoke.py --device cuda
```

`auto` 自动选择 CUDA 或 CPU；明确指定 `cuda` 而设备不可用时会报错。离线 PCA/建表使用 CPU，直接接收恢复和攻击网络支持 GPU；保留的原始图像反演和生成脚本需要 CUDA。

## 文件与数据

- `src/fiber_semantic`：独立 CPU 编解码模块及可微 PyTorch 接收模块。
- `research`：从实际工程选出的发送端损失修正版、门控 TCN、图像生成、评估与计时代码。
- `data/measured`：4 个真实 MATLAB 文件，包含两个温度与重复测量，约 2.5 MB。
- `data/published_metrics`：历史汇总数值，不是本次重新训练产生的结果。
- `docs/VALIDATION.md`：实际执行的验证及其范围。

完整图像实验的依赖和命令见 [research/README.md](research/README.md)。PyTorch、CUDA、大型权重、完整数据集及旧训练输出均不打包。

本项目公开的标定状态仅作为测试样例，不再属于保密密钥。实验采用 **20 km 单模光纤链路**，已由作者确认；数据文件名、示例命令和说明均已统一。已有实验支持固定映射可学习、物理状态切换后的旧攻击模型难以迁移；不等同于证明密码学安全或端到端实时通信。

上传方式见 [docs/PUBLISHING.md](docs/PUBLISHING.md)。

