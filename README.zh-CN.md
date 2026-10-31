<p align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/readme-banner.png" alt="TileOPs × TileFoundry——从内核到算子库：由智能体构建，为智能体服务" width="100%">
</p>

<h3 align="center">跨后端的规格驱动 LLM 算子——由智能体构建，为智能体服务</h3>

<p align="center">
  <i>TileOPs 定义每个算子必须完成的功能。<br>
  编码智能体与专为智能体打造的 AI 编译器 TileFoundry 通过迭代不断扩展算子库。</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/pypi/v/tileops?color=1E90FF" alt="PyPI 版本"></a>
  <a href="https://tile-ai.github.io/TileOPs.github.io/user-guide/manifest/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="规格覆盖率"></a>
  <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json&label=bench" alt="基准测试覆盖率"></a>
  <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs-nightly%2Fsnapshots%2Fnightly-vs-baseline.json&label=vs%20best" alt="最新每夜构建中领先或追平最快基线的记录数"></a>
  <a href="https://github.com/tile-ai/TileFoundry"><img src="https://img.shields.io/github/issues-search/tile-ai/TileOPs?query=is%3Apr%20is%3Amerged%20label%3Afoundry&label=Forged%20by%20TileFoundry&color=0891b2&logo=github" alt="由 TileFoundry 生成的内核"></a>
</p>

<p align="center">
  <a href="#快速开始"><b>快速开始</b></a> ·
  <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><b>基准测试</b></a> ·
  <a href="https://tile-ai.github.io/TileOPs.github.io/"><b>文档</b></a> ·
  <a href="https://github.com/tile-ai/TileFoundry"><b>TileFoundry</b></a>
</p>

______________________________________________________________________

## 智能体与编译器协同工作

| **TileOPs**<br>智能体以规格为准                                                                                                                                                          | **[TileFoundry](https://github.com/tile-ai/TileFoundry)**<br>编译器响应智能体的需求                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 每个算子都始于一份 YAML 规格。测试和基准测试使用规格中声明的参考实现、工作负载和 Roofline 公式来检查正确性并衡量性能。 | 一款基于 Tile 的 AI 编译器，编码智能体可使用它为不同硬件平台生成高性能程序。 |

TileOPs 展示了编码智能体与 TileFoundry 如何通过循环协作来构建算子库。由每个算子的规格驱动的测试和基准测试会报告失败信息以及与 Roofline 性能上界的差距，智能体和编译器再根据这些反馈进行迭代。通过检查的内核会纳入算子库，供后续工作复用和扩展。这种方式支持构建一个持续成长、易于维护的算子库，而非仅有几个一次性的内核。

```text
规格 ──▶ 编码智能体 ◀──▶ TileFoundry ──▶ 内核
              ▲            （AI 编译器）       │
              │                                ▼
              └─ 失败信息、Roofline 差距 ◀── 测试 + 基准测试
                                               │ 通过
                                               ▼
                                        TileOPs 算子库
```

使用 TileFoundry 生成内核的 PR 会带有 `foundry` 标签。上方徽章统计了带有此标签且已合并的 PR 数量。这些 PR 与其他所有 PR 一样，都需要通过相同的 CI 检查。

## 快速开始

```python
import torch
from tileops.gemm import GEMMFwdOp

gemm = GEMMFwdOp()  # 调用时自动推导形状和数据类型

a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 512, device="cuda", dtype=torch.float16)

d = gemm(a, b)  # 等价于 a @ b.T
```

调用 `op.request_tune()` 后，算子会进行自动调优。对于[图测试组](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#cuda-graph-validation)中的调用，H200 上的 CUDA Graph 支持已在预热后通过验证。凡是在清单条目中具有调用时张量输入、且不包含组合操作的算子，均支持 `torch.compile(fullgraph=True)`。

## 算子库如何持续扩展

实现可以根据规格重新生成，但规格无法从实现中还原。智能体面向整个算子库工作，而不仅仅是其中的内核，因此算子库在扩展过程中必须保持一致性。

- **可维护。** 每个算子都始于一份自包含的 YAML 规格，该规格是代码生成的唯一输入。检查机制会确保 Python 算子与 TileLang 内核之间的边界。
- **可验证。** 规格在编写代码之前就声明了正确性和性能标准。CI 会检查规格、生成的代码、测试和基准测试。
- **可调优。** Roofline 模型会报告每个内核与其性能上界的差距，每夜[基准测试](https://tile-ai.github.io/TileOPs.github.io/benchmarks/)则会将每个内核与同一 GPU 上最快的其他实现进行比较。

<details>
<summary><b>深入了解算子规格</b></summary>

每个算子在
[`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec/)
中都有一条对应条目，每个算子系列使用一个文件。以下内容节选自 `gemm.yaml`：

```yaml
GEMMFwdOp:
  ref_api: nvmath.linalg.advanced.Matmul.execute
  family: gemm
  signature:
    forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}
    params: {trans_a: {type: bool, default: false}, trans_b: {type: bool, default: true}}
    inputs: {a: {dtype: T, shape: "Mat[trans_a, M, K]"}, b: {dtype: T, shape: "Mat[trans_b, K, N]"}}
    outputs: {d: {dtype: T, shape: "[M, N]"}}
  workloads: [{M: 1024, N: 1024, K: 1024, trans_a: false, trans_b: false, dtype_cases: [{T: float16}, {T: bfloat16}], label: square-1k}]
  roofline: {flops: "2 * M * N * K"}
```

`Mat[t, R, C]` 是该条目声明的一种类型：其形状为 `[R, C]`；当 `t` 为 true 时，形状为 `[C, R]`。

| 字段        | 作用                                                                              |
| ----------- | --------------------------------------------------------------------------------- |
| `ref_api`   | 在适用的情况下，算子所遵循语义的 API。                                             |
| `signature` | 基于命名索引的张量类型。每次调用在分派前都会依据它进行检查。                       |
| `workloads` | 契约测试和每夜基准测试使用的具体调用。                                             |
| `roofline`  | 性能模型。`eval_roofline()` 会针对每个受检查的调用对其进行求值。                   |

验证器会在 CI 中检查每个条目是否与其实现一致。
[layer-boundaries.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/layer-boundaries.md)
说明了算子各层的职责。

</details>

## 安装

```bash
pip install tileops
```

pip 会从 PyPI 解析并安装 PyTorch 2.13、[TileLang](https://github.com/tile-ai/tilelang) 0.1.12 和 PyYAML。其余环境需由本机提供：

- Linux，以及配备驱动程序、计算能力为 9.0（SM90）的 NVIDIA GPU
- CUDA Toolkit 13.2，并确保 `nvcc` 位于 `PATH` 中：TileLang 会在运行时编译内核

该版本已在 H200 上使用 Python 3.12 和 3.13 完成验证。

部分内核还提供 SM80、SM86 或 SM89 路径。在这些设备上运行时会进行相应测试，但它们不在发布验证范围内。

如需开发 TileOPs，请从源码安装：

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e . --group dev -c constraints.txt   # constraints.txt 固定了 CI 验证的依赖版本
pre-commit install

python -m pytest -q tests -m smoke           # 验证；需要 CUDA GPU
```

预构建的 Docker 镜像包含完整的软件栈，并提供与 CI 一致的环境。有关详细信息、测试层级、基准测试和构建故障排查，请参阅
[development.md](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#dev-docker-image)。

## 文档

- [文档网站](https://tile-ai.github.io/TileOPs.github.io/)：API 参考、用户指南和每夜基准测试。
- [开发指南](https://tile-ai.github.io/TileOPs.github.io/user-guide/development/)：环境、测试和拉取请求。
- [CONTRIBUTING.md](https://github.com/tile-ai/TileOPs/blob/main/CONTRIBUTING.md)：命名、PR 结构和审查标准。
- [设计文档](https://github.com/tile-ai/TileOPs/tree/main/docs/design)：架构、清单、Roofline 和层边界。
- [API 稳定性](https://github.com/tile-ai/TileOPs/blob/main/docs/stability.md)：公共模块、兼容性承诺、支持范围，以及在 TileOPs 之外编写算子。

## 许可证

TileOPs 基于 [MIT 许可证](https://github.com/tile-ai/TileOPs/blob/main/LICENSE)发布。
