# 昇腾 NPU 上的 MFU 计算方法

从 2MNK 到掩码场景的有效计算量。

**适用平台：**昇腾 NPU（以 Ascend 950PR 为例）。

MFU 衡量模型在给定硬件上把峰值算力用出了多少：分子是模型真正需要的浮点运算量，分母是运行时间与硬件峰值的乘积。它反映本文所统计的矩阵乘等浮点运算的利用率，并不等于一般意义的硬件利用率。访存、通信、调度占用的时间都记在分母里，但不增加分子。

下面先给出无掩码 GEMM 的标准算法，再说明掩码场景为什么必须按有效计算量统计。

**核心公式：**

$$
\mathrm{MFU}
= \frac{\text{有效浮点运算量}}{\text{运行时间}\times\text{硬件峰值算力}}
= \frac{\mathrm{FLOPs}_{\mathrm{eff}}}{T\,P_{\mathrm{peak}}}
$$

无掩码 GEMM 的运算量为 2MNK；有掩码时，只统计掩码保留的有效位置。

## 1. MFU 的定义

分子与分母必须采用相同口径：

- **数据类型一致：**FP16、BF16、FP32、INT8 等。
- **稀疏模式一致：**dense 或 sparse。
- **设备范围一致：**单卡或整机。
- **计时范围一致：**单算子对单算子，整步对整步。

昇腾不同型号、不同数据类型的峰值差别很大，口径混用会得到没有意义的数字。

其中，T 为运行时间，单位为秒；P 表示匹配口径的峰值算力。使用 TFLOPS 数据时，应乘以 10¹² 后代入公式。

## 2. 硬件峰值算力

峰值算力取自官方规格或下表。Ascend 950PR 的峰值按启用的 AIC 数量分为 32 AIC 和 28 AIC 两档。

**表 1　Ascend 950PR 技术规格**

### 2.1 核心数量

| 项目 | 32 AIC | 28 AIC |
| :--- | ---: | ---: |
| Cube Core 数量 | 32 | 28 |
| Vector Core 数量 | 64 | 56 |

### 2.2 Cube + Vector 总算力

| 数据类型 | 单位 | 32 AIC | 28 AIC |
| :--- | :--- | ---: | ---: |
| MXFP4 | TFLOPS | 1784 | 1561 |
| HiF8 / MXFP8 / FP8 | TFLOPS | 919 | 804 |
| INT8 | TOPS | 919 | 804 |
| BF16 / FP16 | TFLOPS | 486 | 425 |
| TF32 | TFLOPS | 243 | 212 |

### 2.3 Cube 算力

| 数据类型 | 单位 | 32 AIC | 28 AIC |
| :--- | :--- | ---: | ---: |
| MXFP4 | TFLOPS | 1730 | 1513 |
| HiF8 / MXFP8 / FP8 | TFLOPS | 865 | 756 |
| INT8 | TOPS | 865 | 756 |
| BF16 / FP16 | TFLOPS | 432 | 378 |
| TF32 | TFLOPS | 216 | 189 |

### 2.4 Vector 算力

| 数据类型 | 单位 | 32 AIC | 28 AIC |
| :--- | :--- | ---: | ---: |
| FP16 / BF16 | TFLOPS | 54 | 47 |
| FP32 | TFLOPS | 27 | 23 |
| INT8 | TOPS | 54 | 47 |
| INT16 | TOPS | 27 | 23 |
| INT32 | TOPS | 13 | 11 |
| INT64 | TOPS | 6 | 5 |

**数据来源：**原稿注明为昇腾官网，2026 年 10 月查询。浮点算力单位为 TFLOPS，整数算力单位为 TOPS。本次格式整理保留原稿数据，未重新核验官网规格。

**配置说明：**原稿将 32 AIC / 28 AIC 解释为配置启用的 AI Core（Cube Core）数量，并注明官网仅以 32 / 28 并列，未标注列名。PR 与 DT 是另一组产品线：PR 最高 32 AIC、128 GB、1.6 TB/s；DT 最高 36 AIC、144 GB、4 TB/s。两者同属 950 代，灵衢架构一致。

**本文矩阵乘 MFU 的峰值选取：**

- 分母只取 Cube 算力中与实际执行数据类型对应的一行。FP16 输入配 FP32 累加仍按 FP16 计。
- Cube + Vector 总算力把不参与矩阵乘的 Vector 也加了进来，会使分母偏高约 12.5%，MFU 相应偏低。
- 28 AIC 配置的峰值约为 32 AIC 的 87.5%，应以实际机器读到的规格为准。
- 整机 MFU 的峰值算力为各卡峰值之和。

## 3. 无掩码矩阵乘：FLOPs = 2MNK

最基本的 GEMM 为：

$$
C_{M\times N}=A_{M\times K}B_{K\times N}
$$

输出 C 共有 MN 个元素，每个元素对应长度为 K 的点积。按一次乘加计 2 FLOPs 的惯例，乘加次数为 MNK，总运算量为：

$$
\mathrm{FLOPs}=2MNK
$$

因此：

$$
\mathrm{MFU}=\frac{2MNK}{T\,P_{\mathrm{peak}}}
$$

### 3.1 计算示例

对于 M = N = K = 4096 的 FP16 GEMM，若：

$$
T=1.0\ \mathrm{ms}=10^{-3}\ \mathrm{s}
$$

$$
P_{\mathrm{peak}}=432\ \mathrm{TFLOPS}
=432\times10^{12}\ \mathrm{FLOP/s}
$$

这里使用 Ascend 950PR、32 AIC 配置的 FP16 Cube 峰值，则：

$$
\mathrm{FLOPs}=2\times4096^3\approx1.37\times10^{11}
$$

$$
\mathrm{MFU}
=\frac{2\times4096^3}{10^{-3}\times432\times10^{12}}
\approx31.8\%\approx32\%
$$

### 3.2 多卡与计时范围

多卡时，若报告整机 MFU，分子取各卡 FLOPs 之和，分母中的峰值取各卡峰值之和：

$$
\mathrm{MFU}_{\mathrm{system}}
=\frac{\sum_{j=1}^{N_{\mathrm{card}}}\mathrm{FLOPs}_j}
{T\sum_{j=1}^{N_{\mathrm{card}}}P_{\mathrm{peak},j}}
$$

若报告单卡 MFU，则只取该卡承担的 FLOPs 与该卡峰值。流水线并行（PP）的流水空泡计入 T，但不增加 FLOPs。

按本文口径，分子只统计矩阵乘：LayerNorm、Softmax、通信不计入分子，但其耗时留在对应完整计时范围的 T 中。受 Cube 利用率、tile 形状与访存限制，实测 MFU 通常明显低于 100%。

## 4. 有掩码：按有效计算量统计

掩码会把部分输出位置置为无效，例如 causal attention 的上三角、变长序列的 padding、块稀疏掩码。硬件可能仍按完整 tile 计算，但语义上只有保留位置参与结果，因此分子应采用有效计算量：

$$
\mathrm{FLOPs}_{\mathrm{eff}}=2K\,N_{\mathrm{valid}}
$$

其中，有效元素数 N 为有效输出元素个数。逐行统计为：

$$
N_{\mathrm{valid}}=\sum_i n_i
$$

nᵢ 为第 i 行的有效元素数。按完整有效块统计时：

$$
\mathrm{FLOPs}_{\mathrm{eff}}=\sum_b 2M_bN_bK_b
$$

代回定义：

$$
\mathrm{MFU}_{\mathrm{eff}}
=\frac{\mathrm{FLOPs}_{\mathrm{eff}}}{T\,P_{\mathrm{peak}}}
$$

### 4.1 Causal self-attention 示例

设 Q、K 均为 S × d，则 QKᵀ 的 score 矩阵为 S × S。仅下三角（含对角线）有效时：

$$
N_{\mathrm{valid}}=\frac{S(S+1)}{2}
$$

有效位置占全量的比例为：

$$
\frac{N_{\mathrm{valid}}}{S^2}
=\frac{S+1}{2S}
\approx\frac12\qquad(S\text{ 较大时})
$$

因此，QKᵀ 的有效运算量为：

$$
\mathrm{FLOPs}_{QK^{\mathsf T},\mathrm{eff}}
=2d\times\frac{S(S+1)}{2}
=dS(S+1)
\approx S^2d
$$

当 V 的特征维度同为 d 时，PV 的有效运算量同理，也约为全量的一半。直接套用完整矩阵的 2MNK，会把这一掩码口径下的 attention 有效运算量高估近两倍。

### 4.2 有效口径与执行口径

| 口径 | 分子 | 含义 |
| :--- | :--- | :--- |
| 有效 MFU | 有效运算量 FLOPs | 衡量掩码之后真正有用的计算效率，是本文推荐口径 |
| 执行口径 MFU（原稿称“硬件利用率”） | 全量 2MNK | 在硬件执行完整 tile 的前提下，按全量矩阵乘估算执行运算量 |

按上述前提，两种口径之比为：

$$
\frac{\mathrm{MFU}_{\mathrm{exec}}}{\mathrm{MFU}_{\mathrm{eff}}}
=\frac{\mathrm{FLOPs}_{\mathrm{exec}}}{\mathrm{FLOPs}_{\mathrm{eff}}}
$$

该比例表示掩码带来的算力放大倍数；causal attention 在 S 较大时约为 2∶1。报告时应注明口径。

## 5. 速查表

**表 2　常见场景的运算量**

| 场景 | 运算量（FLOPs） | 说明 |
| :--- | :--- | :--- |
| 无掩码 GEMM，维度为 M、N、K | 2MNK | 一次乘加计 2 FLOPs |
| Causal QKᵀ，Q、K 的形状均为 S × d | dS(S + 1) ≈ S²d | 约为全量 2S²d 的一半 |
| Causal PV，V 的特征维度为 d | dS(S + 1) ≈ S²d | 同上 |
| 任意输出位置掩码，归约长度为 K | 2K × N（有效） | N（有效）= ∑ᵢ nᵢ |
| 整机，N 张同规格、同工作量卡 | 卡数 × 单卡 FLOPs | 峰值同步求和；工作量不同时逐卡求和 |
