# TopKSelectFwdOp Ascend C 实现方案

日期：2026-10-30
性质：设计文档，未经 950PR 编译或实测验证
关联文档：[TileLang 适配方案 v3](./topk-select-ascend-950pr-plan.md)、[复审意见 v2](./topk-select-ascend-950pr-review-v2.md)、[Radix 算法分析](./topk-select-ascend-radix-analysis.md)

## 0. 文档定位

本文描述如何从零用 Ascend C 实现一个**完整满足公共契约**的 `TopKSelectFwdOp` 单算子。

需要明确区分两条路线的定位：

| 路线 | 定位 |
| --- | --- |
| TileLang Ascend | 主管要求的最终交付形态 |
| Ascend C（本文） | 原生基线 / 技术参照 / 算法验证载体 |

本文不改变交付要求。它的作用是：在 TileLang 版本存在工具链不确定性时，先用 Ascend C 把**契约、算法、确定性规则**验证到可运行状态，为 TileLang 实现提供可比对的原生参照，同时可作为性能基线的候选来源之一。

文中所有资源数字均为设计预算，不是设备实测结论。

## 1. 契约冻结

实现前必须冻结以下语义，任何一项在实现中途变更都会导致已完成的验证作废。

| 项目 | 冻结内容 |
| --- | --- |
| 输入 | `logits` FP32 `[B,S,L,G]`；`starts`/`ends` int32 `[B,S]`，同设备 |
| 参数 | `top_k` 为算子属性，`0 < K <= L` |
| 选择域 | 每个 `(b,s,g)` 独立在 `[starts[b,s], ends[b,s])` 内选择 |
| 输出 | 新分配连续 int32 `[B,S,G,K]`，绝对 KV 索引 |
| 槽位顺序 | 无序；两次调用槽位排列可不同 |
| 确定性 | 同一输入跨调用选中**索引集合**相同，含阈值同分 |
| 短窗口 | 有效位置数小于 K 时全部返回，其余槽位填 `L` |
| 空/反向窗口 | `starts >= ends` 时全部填 `L`，不新增报错 |
| 存储 | 输入不变，输出不与输入别名，有效索引唯一 |
| 编译 | eager 与 `torch.compile(fullgraph=True)` 均须通过 |
| 输入域 | 不排除 NaN、Inf、±0；不得整体降精度 |

### 1.1 确定性全序

统一采用：有效位置优先 → 分数降序 → 同分按绝对索引升序。

- `+0` 与 `-0` 视为同分；
- NaN 统一为最高键，不同 payload 视为同分；
- 不给分数加 epsilon 破平局，避免改变极接近的 FP32 分数先后。

NaN 的最终优先级仍需以 CPU 参考实验为准；在实验关闭前，该规则标记为**待确认**，不视为已验证事实。

## 2. 工程结构

```
topk_select/
├── op_host/
│   ├── topk_select_def.cpp          # 原型定义、dtype、ascend950 注册
│   ├── topk_select_infershape.cpp   # 输出 shape [B,S,G,K]
│   ├── topk_select_tiling.h         # TilingData 字段
│   └── topk_select_tiling.cpp       # 按 UB 预算计算分块参数
├── op_kernel/
│   └── topk_select.cpp              # 入口 + KernelTopKSelect
├── op_api/
│   ├── aclnn_topk_select.h
│   └── aclnn_topk_select.cpp        # aclnnTopkSelectGetWorkspaceSize / 执行
├── op_graph/
│   └── topk_select_proto.h          # 图模式 proto（后置）
└── tests/
    ├── golden.py                    # CPU 全序参考实现
    ├── test_contract.py             # 契约用例
    └── bench.py                     # 延迟与 MFU 采集
```

### 2.1 原型定义要点

```cpp
class TopKSelect : public OpDef {
public:
    explicit TopKSelect(const char* name) : OpDef(name) {
        this->Input("logits").ParamType(REQUIRED)
            .DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("starts").ParamType(REQUIRED)
            .DataType({DT_INT32}).Format({FORMAT_ND});
        this->Input("ends").ParamType(REQUIRED)
            .DataType({DT_INT32}).Format({FORMAT_ND});
        this->Attr("top_k").AttrType(REQUIRED).Int();
        this->Output("indices").ParamType(REQUIRED)
            .DataType({DT_INT32}).Format({FORMAT_ND});

        AICoreConfig cfg;
        cfg.DynamicCompileStaticFlag = true;
        this->AICore().AddConfig("ascend950", cfg);
    }
};
OP_ADD(TopKSelect);
```

`ascend950` 必须显式注册；不注册时算子不会进入该 SoC 的调度表。

### 2.2 Kernel 入口

```cpp
extern "C" __global__ __aicore__ void topk_select(
    GM_ADDR logits, GM_ADDR starts, GM_ADDR ends, GM_ADDR indices,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelTopKSelect op;
    op.Init(logits, starts, ends, indices, tilingData);
    op.Process();
}
```

`workspace` 在 V1 设计下不被使用（见 D1），保留形参以兼容 aclnn 接口约定。

## 3. 关键设计决策

### D1：一行 `(b,s,g)` 由一个 AI Core 块独占，不做跨核合并

**决策**：块通过 grid-stride 领取行号，`row = GetBlockIdx(); row < rows; row += GetBlockNum()`。

**理由**：
- 每行的 TopK 相互独立，天然无跨核依赖；
- 无需全局 workspace，也无需跨核屏障；
- 结果与核数、调度顺序无关，直接满足跨调用确定性。

**被否方案**：一行拆给多核再归并。该方案需要 workspace、需要层间依赖，且归并阶段本身要再实现一次确定性合并，复杂度翻倍。

**触发重新评估的条件**：单行最长窗口在 UB 中无法用固定分块走完，或实测显示行间负载极不均衡导致核利用率低于阈值。届时升级为 V2，并保留 V1 作为正确性对照。

### D2：`top_k` 作为属性，不作为输入张量

**理由**：输出 shape `[B,S,G,K]` 依赖 K；作为张量会导致 shape 在 host 侧不可推导，破坏 infershape 与图模式。

### D3：两级键设计——FP16 高位粗筛 + FP32 完整键精确细化

**决策**：
- Stage A 用 FP16 有序编码的高 8 位构造 256 桶直方图；
- Stage B 用 FP32 位变换后的完整 32 位键逐字节细化。

**理由**：粗筛把数据量压到常数级候选，精化只沿第 K 名所在桶向下追踪，不需要对整行排序。FP16 只参与分桶，最终选中由 FP32 完整键决定，因此不构成精度降级。

**约束**：必须验证“非 NaN 分数的粗桶编号随精确键单调不减”，重点检查 FP16 舍入边界、上溢、下溢、±0、±Inf。若所选转换不满足单调性，退回直接对 FP32 键做 4 轮 radix（正确性对照），但两者的性能数据不可混用。

**被否方案**：直接用 4 轮 32 位 radix 覆盖全窗口。正确但多一轮全量扫描，语义上没问题，性能上更差，保留为 fallback 和对照。

### D4：候选缓存有界，溢出时重扫 GM 窗口

**决策**：候选缓存容量固定（如 2048 条 × (4B 索引 + 4B 键)）。总数始终完整统计；缓存装不下时，下一轮重新遍历窗口，用粗桶与已确定的前缀过滤。

**理由**：物化全行候选在 L=65536、K=2048 时不可行；有界缓存是空间可证的唯一方式。

**必须保留的验证用例**：8192 个分数挤在同一粗桶、窗口越过桶缓存边界。这两类用例专门打穿“只在前 2048 条里继续选”的错误实现。

### D5：同分按绝对索引升序，用块内前缀和补足

**决策**：最后一轮确定阈值键 `t` 后，先输出所有 `key > t`，再按绝对索引升序取 `key == t` 的前 `r` 个；写入位置由块内前缀和决定。

```cpp
int32_t prefix = BlockPrefixCount(equalMask);   // 本线程之前有多少个同键候选
if (isTie && prefix < remain) {
    outGm_.SetValue(pos + prefix, absIdx);
}
```

**理由**：`atomic_add` 抢位只能保证写入地址唯一，不保证较小索引优先，跨调用选中集合会漂移。现有 GPU 实现与官方 `AscendC::TopK` 都未提供该保证，因此这一层必须自研。

**成本**：前缀和是对 `C` 个布尔值的块内扫描，量级为 `O(log C)` 步同步，只在最后一轮执行一次。

### D6：不把 `AscendC::TopK` 作为最终选择器

**决策**：可以尝试用它加速“块内前 K”的局部选择，但最终阈值同分处理必须自研；若其约束无法满足，直接使用自研 radix 路径。

**理由**：
- 其同分规则不可指定，无法满足 D5；
- 3510 归并路径存在 `inner <= 4096`、`1 <= k <= inner`、`inner` 为 32 倍数等约束，与主规格 K=2048 的分块设计不一定兼容；
- 它是 device API，不接收 GM 分数与动态窗口，不构成端到端算子。

**澄清**：此前观测到的“官方 TopK 头文件在 TileLang 编译回调下成员重复声明失败”，是 TileLang 生成代码与 SDK 头文件混编的问题；纯 Ascend C 工程走 CMake + bisheng 不受该问题约束，不能据此判定 SDK 不可用。

### D7：先全填 `L`，再覆盖选中槽

**决策**：每行进入计算前，先把输出 `[K]` 全部写 `L`，后续只写入真正选中的索引。

**理由**：短窗、空窗、反向窗三种情况都不需要额外分支，且避免未初始化内存泄漏到输出。这是契约的一等要求，不是兜底技巧。

### D8：Python 侧用 `custom_op` + fake impl 接入

```python
@torch.library.custom_op("tileops_npu::topk_select", mutates_args=())
def topk_select(logits: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor, top_k: int) -> torch.Tensor:
    return torch.ops.npu.topk_select(logits, starts, ends, top_k)

@topk_select.register_fake
def _(logits, starts, ends, top_k):
    b, s, _, g = logits.shape
    return torch.empty((b, s, g, top_k), dtype=torch.int32, device=logits.device)
```

**理由**：`torch.compile(fullgraph=True)` 依赖 fake impl 推导 shape/dtype；缺失会导致 graph break，被误判为编译失败。

### D9：Tiling 只给固定预算，窗口在设备侧读取

**决策**：host tiling 输出分块大小 `C`、缓存容量、每核行数等固定参数；`starts/ends` 只在 kernel 内读取并动态控制循环。

**理由**：窗口边界是运行时数据，host 不可见。任何按宽度分组的负载均衡都需要额外统计 kernel，V1 不做。

**代价**：行间负载不均衡无法在调度层缓解，靠 grid-stride 自然摊平。

### D10：分块 `C` 从 2048 起步

**决策**：首版 `C = 2048`，验证通过后再评估 4096。

**UB 预算（设计值，非实测）**：

| 项 | 大小 |
| --- | --- |
| 分数分块 `C × 4B` | 8 KiB |
| 有序键分块 `C × 4B` | 8 KiB |
| 直方图 256 × int32 | 1 KiB |
| 候选缓存 `2 × 2048 × 8B` | 32 KiB |
| 输出缓冲 `K × 4B`（K=2048） | 8 KiB |
| 合计 | 57 KiB |

在 128 KiB 设计预算下留有约 2 倍余量，用于编译器临时分配与流水缓冲。`K=2048` 时输出缓冲必须单独计入，不能漏算。

### D11：本方案不做的事

- 不做整行排序；
- 不做 bitonic 选择/合并作为主路径；
- 不引入跨核 workspace 归并；
- 不通过降精度或近似匹配率放行；
- 不使用多算子拼接模拟完整语义。

## 4. Kernel 处理流程

```cpp
__aicore__ inline void ProcessRow(int64_t row) {
    int32_t s = startsGm_.GetValue(row);
    int32_t e = endsGm_.GetValue(row);

    FillOutput(row, L);                 // D7
    if (s >= e) return;                 // 空窗 / 反向窗
    int32_t width = e - s;
    if (width <= K) {                   // 短窗
        WriteAllValid(row, s, e);
        return;
    }
    uint32_t thrBin = BuildCoarseHistogram(row, s, e);
    if (!SelectAboveThreshold(row, s, e, thrBin)) return;
    RefineByBytes(row, s, e);
    FillTiesByIndexOrder(row, s, e);    // D5
}
```

地址映射：

```
logits 元素 = ((b * S + s) * L + kv) * G + g
输出元素   = ((b * S + s) * G + g) * K + slot
```

`logits` 沿 KV 轴的跨度为 `G`，非连续；分块搬运时按 `C × G` 的一次性 stride 读取或按元素 gather，需在实现中明确取舍并计入带宽。

## 5. 里程碑计划

不设时间估计，只设交付物与退出标准。每一级未通过退出标准不得进入下一级。

| 里程碑 | 交付物 | 退出标准 | 阻塞项 |
| --- | --- | --- | --- |
| M0 契约基线 | `golden.py`、`test_contract.py` | 覆盖短窗、空窗、反向窗、全同分、NaN/Inf/±0；golden 自身通过人工核对用例 | 无 |
| M1 工程骨架 | 原型、infershape、tiling、kernel 空壳 | 空窗、短窗、`width<=K` 三类用例与 golden 完全一致；输出全填 `L` 正确 | ascend950 注册可被调度 |
| M2 粗筛 | Stage A 直方图 + 后缀和 + 阈值 | 随机有限值下选中索引集合与 golden 一致 | FP16 粗桶单调性验证通过 |
| M3 精化 | Stage B 四轮字节细化 + 溢出重扫 | 稠密同分、单桶 8192 元素、窗口越界缓存用例通过 | 候选缓存容量与重扫路径正确 |
| M4 确定性 | 索引升序同分补足 | 同一输入连续 100 次调用，输出索引集合逐元素相同 | 前缀和实现正确 |
| M5 接口集成 | aclnn 两级接口 + `custom_op` + fake impl | eager 与 `torch.compile(fullgraph=True)` 结果一致且无 graph break | 图模式 shape 推导 |
| M6 性能 | `bench.py` + profile 记录 | 在同一计时范围、同一输入分布下产出可复现延迟；与基线口径一致 | 基线选定、MFU 峰值口径批准 |

### 5.1 里程碑之间的硬依赖

- M2 不能在 M1 之前做：没有正确的填充与短窗分支，粗筛结果的正确性无法判定。
- M4 不能在 M3 之前做：同分补足需要阈值键已正确定位。
- M5 不能跳过 M4：图模式只验证接口一致性，不验证确定性。
- M6 的最后一步依赖外部条件（基线算子选定、峰值口径批准），不满足时只能产出延迟数据，不得给出达标结论。

## 6. 验证矩阵

| 用例类别 | 内容 | 判定方式 |
| --- | --- | --- |
| 基础 | 全窗口、随机有限值 | 索引集合完全相等 |
| 短窗 | `width < K` | 有效位置集合相等，其余槽位为 `L` |
| 空窗 | `starts == ends` | 全部为 `L` |
| 反向窗 | `starts > ends` | 全部为 `L` |
| 全同分 | 窗口内分数完全相等 | 选中集合等于最小 K 个绝对索引 |
| 稠密同分 | 8192 元素同桶 | 集合相等且跨调用稳定 |
| 边界桶 | 分数恰好落在 FP16 舍入边界 | 集合相等 |
| 特殊值 | NaN、+Inf、-Inf、±0 | 分类型核对，不放松数值容差 |
| 确定性 | 同输入重复调用 | 索引集合逐元素相同 |
| 规模 | K=1024、K=2048、L=65536 | 全部通过，无资源越界 |
| 图模式 | `torch.compile(fullgraph=True)` | 与 eager 完全一致 |

验证原则：**比较索引集合，不比较分数多重集**。现有 workload 只比较选中分数，无法发现同分索引漂移，不能作为本算子的正确性判据。

## 7. 风险与未决项

| 项 | 说明 | 关闭方式 |
| --- | --- | --- |
| UB 实测预算 | 文中 57 KiB 为设计值 | 编译报告 + 实际 UB 占用 |
| 粗桶单调性 | FP16 转换在边界行为未验证 | M2 前单独实验 |
| NaN 全序 | 优先级待参考实验 | CPU/NPU 行为对比 |
| 前缀和成本 | 块内扫描开销未测 | M4 阶段 profile |
| 行间负载不均 | grid-stride 只能摊平 | M6 profile 中评估 |
| 基线与 MFU 口径 | 尚无完整同语义基线 | 由需求方确认后锁定 |

## 8. 与 TileLang 路线的关系

本方案的产出可以在三个地方被 TileLang 路线直接复用：

1. M0 的 golden 与契约用例，与后端无关，可共享；
2. M4 的确定性规则与实现方式，是 TileLang 版本必须解决的同一问题；
3. M6 的实测延迟，可作为原生参照，用于判断 TileLang 版本是否达到可接受水平。

但本方案**不替代** TileLang 交付，也不构成 TileLang 性能达标的证明。
