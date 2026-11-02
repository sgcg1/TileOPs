# TopKSelectFwdOp 昇腾 950PR 适配交接

> 2026-11-02 按4.2.2第4项新增桶扫描候选：见 [scan-r1说明](topk-select-950pr-scan-20261102.md)。代码与现场验证工具已准备；新候选NPU正确性与性能尚待实测，未正式发布v0.1.4。

> 后续同步方式已更新：见 `docs/topk-select-950pr-sync.md`。个人 fork main 管理代码，B2Y `topk-950pr/` 提供手动传入 yjl 容器的辅助工具。以下远端路径和 dirty 状态为交接时历史快照；候选已保存为 Git 提交，但仍未通过 v0.1.4 最终验收。

> 2026-10-31 卡 0 恢复核查：见 `docs/design/topk-select-950pr-card0-resume-20261031.md`。个人 fork 候选在独立缓存下通过 14 项测试，已保存同轮 Decode 基线、实际二进制和负载频率证据；完整 Prefill 性能验收仍未完成。

交接日期：2026-10-31。本文面向完全没有本会话上下文的新会话。

**首先看结论：v0.1.0～v0.1.3 已有提交；v0.1.4 仍是未提交候选，没有完成最终验收。最新同进程对照发现历史性能不可直接外推，下一步优先核实当前基线与运行条件，不能继续盲目叠加优化。**

## 1. 我们在做什么任务

用户负责将 TileOPs 的 `TopKSelectFwdOp` 适配到昇腾 950PR NPU，并优化性能。

- 最终交付必须使用 **TileLang 主仓的 Ascend 后端**。独立的 `tilelang-ascend` 仓不在开发范围内；Triton 不是本任务的交付或对比目标。
- 用户要求优先与已有原生 Ascend C 算子比较。经过多轮讨论，最新明确指定使用 **官方 CANN TopKV2** 做受限规格下的性能基线。
- 原生基线不是自行实现的 TopK，也不是给 `AscendC::TopK` 随意封装的自定义算子。我们构建的 GE 静态模型只是调用官方 TopKV2 的测量入口。
- 用户要求先保存初版，再按“直方图／扫描 → 粗筛 → UB 搬运 → 调度”顺序实验；每轮保存提交、版本号和真实性能，包括失败试验。
- 用户不喜欢反复确认：授权范围内继续工作即可；不要修改共享环境，不要上传或推送代码到远端服务。
- 面向用户的说明和交接文档用中文；项目代码、注释及工具脚本保持英文。

### 1.1 最新性能表达规则

**性能倍数 = Ascend C 基线耗时 / 当前实现耗时。**

- 当前耗时是基线的 10 倍，性能写为 **0.1× Ascend C**。
- 与基线相同为 **1.0× Ascend C**；大于 1 才表示更快。
- 同时保留实际延迟、shape、K 和计时范围。不要再把“耗时倍数”写成“性能倍数”。
- 当前不以 MFU、Vector MFU 或有效带宽效率作为已确定的验收公式；不能沿用早期讨论中的 95% 来宣称达标。

### 1.2 算子接口和语义

输入为 FP32 `scores[B,S,L,G]`、int32 `starts[B,S]` 与 `ends[B,S]`，K 为属性。每个 `(b,s,g)` 在窗口 `[start,end)` 内独立选择，输出新分配、连续的 int32 `[B,S,G,K]`，存绝对 KV 索引。

- 输出槽位无序；同分选择较小绝对索引，要求选中集合确定。
- 短窗口返回全部有效位置，其余填 `L`；空／反向窗口全部填 `L`。
- 当前实现及 CPU golden 把 NaN 排在正无穷之前、统一 NaN payload、把 ±0 视作同分。
- 不改变输入，不通过 FP16 近似结果放行；粗桶可以用 FP16，最终决定必须来自原始 FP32 完整键。
- K 不受 2048 条候选缓存限制，K>2048 必须正确。

## 2. 已经完成了什么

### 2.1 官方性能基线及指标

基线是 **CANN TopKV2**，使用私人 CANN 9.3.0 Weekly，设备 `Ascend950PR_9579`。正式历史测量日期为 2026-10-30。

| 规格 `[B,S,L,G]` | K | TopKV2 纯内核中位数 | 静态模型设备事件中位数 |
| --- | ---: | ---: | ---: |
| `[1,1,16384,1]` | 2048 | **8.721 μs** | **16.207 μs** |
| `[1,32768,65536,1]` | 1024 | **50.652891 ms** | **50.646214 ms** |
| `[1,32768,65536,1]` | 2048 | **52.075164 ms** | **52.088139 ms** |

每组为 50 次非 profiling 事件采样，另行 50 次 profiling 采样，均取中位数。两列来自独立测量，不能相减精确推算开销。事件包围 `aclmdlExecuteAsync`，包括静态模型控制及可能的 host 提交空隙；**纯内核对比用内核列**。

落实的测试条件：

1. G=1、全窗口，相同 FP32 输入、shape、K。
2. `largest=true, sorted=false, indices_dtype=INT32, sort_policy=0, dim=-1`。
3. 每个 shape/K 独立静态编译 `.om`，K 是编译期常量，执行时只有分数输入。
4. 输入、values、indices、数据集及事件预分配；校验后重新 warmup。
5. 不包含 H2D、编译、随机数生成、哈希和结果校验。
6. profiling 每组只发现 TopKV2 计算任务，类型 `KERNEL_MIX_AIV`；无 AICPU 计算、Cast、Transpose。56 blocks 是启动数，不能据此认定 Decode 使用了 56 个有效核。
7. 默认 GE 精度策略曾导致 FP16 量级舍入，已排除。最终构建显式使用 `ge::ir_option::PRECISION_MODE=must_keep_origin_dtype`。

全输入 SHA-256：

- Decode：`c59b9485e9750eaf04d4668934c0383510a15b2cc9365ef9c6c4b9eaf2bf7593`。
- 两组完整 Prefill：`0c435edd506b44a1cc41ea19a140811275f88c0129da89fb8ecb95e96e48e229`，两组 K 的输入相同。
- 初版的 128 行小规模测试不是 32768 行主规格，且没有同规格 CANN 基线；不能直接换算其“×Ascend C”。

原始记录：`artifacts/cann-topkv2-static-20261030/README.md`、三组 `*-timing.json`、`profile-summary.json`、原始 profiler CSV。远端同名目录在 `tileops-work/evidence` 下，保留 `.om`、构建源码及日志。

### 2.2 Git 与已保存版本

本地原来没有 `.git`，不是 PyCharm 隐藏了已有历史。已从当前文件快照初始化本地版本库，没有恢复或伪造官方历史。

- 工作目录：`D:\04_Project\TileOPs`。
- 分支：`npu/topk-select-950pr`。
- `.idea/vcs.xml` 已设置项目根目录 Git 映射，该文件被原有 IDE ignore 排除，但本机存在。
- `/artifacts/` 已加入 `.gitignore`，避免提交缓存、SDK 和大型 trace。
- Git 本仓库 `core.autocrlf=input`。没有推送远端，也没有配置/恢复官方 upstream 历史。

以下均为**历史批次纯内核中位数**。表中的性能倍数只按历史 CANN 内核列计算，不能当成尚未复测的当前运行条件结论。

| 版本／实现提交 | 具体改动 | Decode | Prefill K1024 | Prefill K2048 | 128 行小规模 | 验证与处置 |
| --- | --- | --- | --- | --- | --- | --- |
| `topk-npu-v0.1.0` / `2dfa070` | 初版；4 轮 FP32 radix，SIMT 原子直方图，串行 256 桶扫描，2048 条双缓冲候选缓存，最多 32 workers | 213.613 μs；**0.0408× Ascend C** | 592.683430 ms；**0.0855×** | 1046.961737 ms；**0.0497×** | 3.642353 ms | 初始 9 项 NPU 测试通过；初版回退点 |
| `topk-npu-v0.1.1` / `cf498c9` | 配额归零后跳过最终同分扫描；保存之前 SIMD／warp 负试验 | 394.5915 μs，profiling 异常，不作为可靠倍数 | 574.360500 ms；**0.0882×** | 1067.905979 ms；**0.0488×** | 3.635645 ms | 10 项通过；experiment，未确认整体收益，后续没有叠加该 guard |
| `topk-npu-v0.1.2` / `cb01af6` | FP16 高字节粗桶 + 4 轮原始 FP32 精化；粗筛时只为需要缓存的候选生成完整键 | 196.528 μs；**0.0444×** | 736.264749 ms；**0.0688×** | 798.202121 ms；**0.0652×** | 2.585055 ms | 14 项通过；K2048 改善、K1024 回退，不适合无条件覆盖初版 |
| `topk-npu-v0.1.3` / `1db546f` | 从 v0.1.2 增加 G=1、L≤16384 的整行 UB 缓存，其余路径回退；最大显式 UB 约 97.5 KiB | 3007.7485 μs | 未测 | 未测 | 44.894853 ms | 14 项通过；当时与历史数据相比明显回退，停止大规模跑分并拒绝采用。新对照显示跨批次归因不可靠，见第 3 节 |

补充说明：

- 初版另一次 50 条 profiling 的 Decode 值为 213.275 μs；上表使用后来统一脚本记录的 213.613 μs，不把两次混成同一次。
- v0.1.1 的非 profiling Decode 事件中位数为 212.382 μs，与 profiling 394.592 μs 明显不一致。不能据此宣传性能收益。
- v0.1.2 的 K2048 事件中位数为 758.854 ms，与 profiling 798.202 ms 不同；128 行事件值 3.745604 ms 也与纯内核不同。原始数据全部保留。
- v0.1.3 没有测完整 Prefill，绝不能把 128 行结果按行数线性放大当成主规格结果。

每个已提交版本的完整样本、输入哈希、源码归一化 SHA-256 和 Git blob SHA-1 在 `docs/performance/topk-select-950pr/v0.1.*.json`。过程和负试验说明在 `docs/design/topk-select-optimization-history.md`。

### 2.3 服务器环境与权限边界

| 项目 | 当前已知信息 |
| --- | --- |
| SSH | `123.60.230.63`，用户 `f00959234` |
| 密码 | 用户此前提供；**不写入交接文档或 Git**。无上下文会话需向用户获取，凭标准输入传给 helper，禁止硬编码 |
| 唯一允许写入／影响的文件环境 | **`/home/f00959234/yjl`**；不能改父目录 `/home/f00959234`，也不能改公共 SDK、驱动、全局 Conda 或其他用户环境 |
| 远端项目 | `/home/f00959234/yjl/tileops-work/TileOPs` |
| 激活脚本 | `/home/f00959234/yjl/tileops-work/activate.sh` |
| Conda 环境 | `/home/f00959234/yjl/envs/tileops-p1-native` |
| Python / torch / torch_npu | 3.12.16 / 2.13.0+cpu / 2.13.0；`+cpu` 标签不代表 NPU 不可用，需导入 torch_npu |
| TileLang | 主仓路线的私有 0.1.15 安装；准确上游 commit 仍未可靠追溯，不能凭版本号编造 commit |
| CANN | `/home/f00959234/yjl/toolchains/cann-9.3-weekly-20261021/cann-9.3.0`；安装信息 9.3.0，`V100R001C13B023`，x86_64-linux |
| 芯片 | `Ascend950PR_9579`，28 Cube / 56 Vector，约 125977 MB 内存、128 MB L2（此前设备查询） |
| 编译平台资源查询 | `UB_SIZE=253952` B（248 KiB），`CORE_NUM=28`；后者不是 Vector 核数。属于编译规格，不等同于实际 kernel UB 分配报告 |
| 测试设备 | `ASCEND_RT_VISIBLE_DEVICES=1`，**物理 1，进程内逻辑 0**；pytest 用 `--tileops-device=npu:0` |

工具包安装版本不能证明所有内部组件版本相同。不要看到旧组件查询显示 9.2 就擅自升级公共驱动或修改共享安装。

所有缓存、tmp、日志、编译输出必须位于 yjl。现有 `base.sh` 已设置 `TMPDIR=/home/f00959234/yjl/tmpko`、XDG/CANN/TileLang/Torch 缓存、`ASCEND_PROCESS_LOG_PATH`、`ASCEND_WORK_PATH`、`PYTHONDONTWRITEBYTECODE=1`、`TASK_QUEUE_ENABLE=0` 和 `ulimit -c 0`。不要自行把 HOME 重定向成临时目录。

## 3. 当前卡在哪

### 3.1 最新关键证据：跨批次发生显著延迟漂移

最后完成了 `control-v2.py`：同一进程、同一预分配输入与输出、相同设备，v0.1.2 与当前 v0.1.4 的 128 线程配置做 AB/BA 交替、50 次设备事件采样，结果集合相同。

| 对照 | 本轮设备事件中位数 |
| --- | ---: |
| v0.1.2 控制版本 | **3237.674 μs** |
| 当前 v0.1.4，COARSE=True、H=128 | **3243.436 μs** |

两者生成 C++ 源码的 SHA-256 **完全相同**：

`40f100f3572376a4099ce4eebab385347b664e2b48b7c4f15cede663b42d70cc`

输入 SHA 仍为上述 Decode 的 `c59b…7593`。

这说明本轮两者的源代码级执行形态一致、时间也接近，却与历史 v0.1.2 事件 201.627 μs 相差显著。**目前不能把近期异常全部归因于 UB 缓存、H 参数或某一优化；也不能据此认定一定是硬件、驱动或编译器的某一个问题。** 尚未比较实际加载二进制、编译参数、库路径、当前设备状态和当前 CANN 基线。

证据：`artifacts/topk-optimization-20261030/v0.1.4/control-v2.json`。远端同版本目录还保留 `control-v2-source.cpp`、`control-v4-source.cpp`，是自行生成的内核源码。

新会话应先解决测量可比性，必要时重新测 CANN 及初版的同轮对照；在此之前，不将当前值直接换成可靠的“×Ascend C”或跨批次加速结论。

### 3.2 v0.1.4 当前状态

**尚未提交、尚未打 v0.1.4 标签，也没有跑最终完整 14 项测试或三组主规格。**

候选改动：

- 已移除 v0.1.3 的 UB 行缓存，回到 v0.1.2 基础。
- kernel 增加编译期 `COARSE: bool=True`、`H: int=128`；H 支持 64/128/256。
- host 读取实际 `vector_core_num`，workers 取 `min(rows, vector_cores)`，本机最多 56。
- host 对 `(L,K)=(65536,1024)` 选直接 FP32 radix，其余仍选粗筛；这是依据现有测试规格作出的经验选择，未证明普遍最优。
- host 的 threads **仍是 128**；不能误称已经采用 256。
- 测量脚本支持 `--threads`，已移除之前未实际使用的 `--chunk`，并记录实际源码路径、生成源码哈希。

已完成的 v0.1.4 调度候选，均为本轮数据，只通过随机有限输入的全行范围／唯一性／阈值证书，不等同于完整契约测试：

| 候选 | 纯内核中位数 | 设备事件中位数 |
| --- | ---: | ---: |
| Decode H=64 | 5185.101 μs | 5200.531 μs |
| Decode H=128 | 3216.976 μs | 3240.569 μs |
| Decode H=256 | 2338.660 μs | 2366.346 μs |
| 128 行、W=32、H=128 | 43.965344 ms | 44.924107 ms |
| 128 行、W=56、H=128 | 34.628275 ms | 34.607901 ms |

同一轮内部 H256/W56 较快，但不能直接拿它们与历史 196 μs／2.585 ms 比较得出因果结论。

当前本地未提交文件恰为：

1. `src/tileops/_kernels/attention/topk_select_ascend.py`
2. `src/tileops/ascend/topk_select.py`
3. `scripts/ascend_topk/measure.py`

已部署 kernel 原始 SHA-256：`b7a00ac6fff52d12982ef0661974aaea59e98af470b1b4ca149634eb91b55d8b`；host 原始 SHA：`c866bef8b87d6cad053a0a32101a12c3919e5e0e99d8bcf9886d99606ff731f4`。部署清单及旧文件备份在远端 `v0.1.4/deployment.json` 与 `before-*.py`。

结束交接前已查询私人目录进程：**没有已知仍在运行的测试、调优或 AB/BA 任务**。仅有状态查询脚本本身。不要依赖旧 exec session ID，下一会话再次核实。

## 4. 下一步计划是什么

按下面的依赖顺序推进，不跳过测量可信性检查：

1. **核实现场和源码。** 阅读第 6 节；检查 Git dirty 文件、远端部署 SHA、实际 Python 模块加载路径、候选进程。不要 reset 或覆盖未提交 v0.1.4。
2. **先重建当前运行条件下的可信对照。** 重复控制实验，比较实际生成代码、编译参数及加载库／二进制哈希；只读检查设备身份与负载。若要重新测 CANN，复用已有静态模型，严格保持第 2.1 节条件。不得为了“固定性能”修改共享设备时钟、驱动或系统设置。
3. **确认后再选调度。** 对 H128/H256、W32/W56 作同进程或相近时段的固定输入对照，保留实际原始样本。不要直接因一次最小值就改默认。
4. **完整验证最终候选。** 运行 14 项契约测试，含 NaN/Inf/±0、舍入边界、上溢／下溢、全同分、缓存溢出、K>2048、非连续输入、分组及空／反向窗口。`torch.compile` 目前只证明 `backend="eager", fullgraph=True`，没有证明默认 Inductor 或 NPU graph replay。
5. **运行四组统一测量。** 三组主规格加 128 行诊断规格；原始分数完全相同，预分配，校验后 warmup，独立事件／profiling。最新脚本为 profiler 启动增加 10 次余量，再按设备时间顺序固定取最后要求数量的真实记录。最终计划为 50 次事件 + 50 条有效 profiling 样本。
6. **记录并提交。** 四组源码哈希必须一致，保存 `docs/performance/topk-select-950pr/v0.1.4.json`、更新历史文档，再提交并打标签。没有整体收益就标为试验／拒绝，最终工作区恢复最佳已验证实现；不制造成功结论。
7. **后续更深层优化。** 在计时可信后，再研究纯 SIMD 键转换／直方图、批量候选压缩、真正的长行 UB 流式分块及 Decode 行内并行。先验证主仓降低能力、同步、scratch 与索引确定性，再做规模测试，不直接照搬 CUDA warp 思路。

关于四轮既定优化的交付状态：统计／扫描试验已记录；粗筛已实现并验证；UB 试验正确但未采用；调度仅完成候选对照，最终验证和版本提交仍未完成。**当前没有达到或证明超过 CANN 的性能目标。**

### 4.2 后续优化方案规划（基于 TopKV2 源码调研的迁移）

> **2026-11-02复评：以下保留交接时的方案与顺序。** 最新Decode阶段数据已定位首轮histogram/compact占已测区间约81%，当前默认也已改为grouped scan；本例没有缓存溢出。后续请以[优化路线复评](topk-select-optimization-roadmap-20261102.md)为准，其中修正了尾部收益、原子与确定性的边界，并给出首轮分片、粗键缓存和Decode首轮分核方案。

> 本节把 CANN 官方 TopKV2 源码（`artifacts/topkv2-analysis-20261030/official/`）的性能手段，映射到当前 TileLang 实现。TopKV2 的优化是三层结构：规模分档选算法 → 选中算法内逐轮收窄 → 多核摊平行负载；其中"逐轮收窄"收益最大，"多核归并"与我们的确定性契约冲突最大。**任何改动都排在解决第 3 节跨批次漂移之后，并逐条对照第 5 节已踩坑，先有同轮对照再进版本。**

#### 4.2.1 迁移总纲

| TopKV2 手段 | 源码依据 | 能否迁移 | 说明 |
| --- | --- | --- | --- |
| 逐轮收窄输入范围 | `topkValueInput_ -= boundaryBinPrevCuSum` | 可（已实现） | v0.1.4 已用边界桶候选缓存实现，见 4.2.2 第 1 项 |
| UB 复用不重拷 | `CopyDataInWithReuseBuffer` | 可（已实现） | v0.1.4 `cached_key/cached_index` 双缓冲已实现，见 4.2.2 第 1 项 |
| 按 k/n 比值切换算法 | `IsFp32MergeSortMode`（`k/n∈[0.25,0.5)`） | 部分可 | 主规格 k/n≈1.5%~3%，纯选择方向正确；仅高占比窗口再重估 |
| 一行拆多核 + 归并 | `lastDimNeedCore=min(maxCoreNum,lastDimTileNum)` | 慎用 | 直指 Decode 单行瓶颈，但归并必须保留同分取小索引 |
| 跨核原子直方图 | `SetAtomicAdd` + `SyncAll` | 直方图可，选索引不可 | 计数可交换故原子安全；"选出具体索引"绝不能用抢位（坑 12） |
| 64 位 value+index 打包 | `sortWithIndex`（`needSortWithIndex_`，K>2000） | 借鉴不照搬 | 一次解决同分排序，排序成本高，仅作超大单桶兜底 |
| 尾轮核利用率阈值 | `LAST_LOOP_CORE_UTILIZATION=0.7` | 可借鉴 | 行数不整除核数时尾轮空转 |
| UB 预算经验值 | `TMP_DATA_NUM=7680`、`SMALL_TILE_DATA_NUM=1024`、`CONST_SIMT_SPACE=32768` | 参考不照搬 | 均为 Ascend C 抽象，TileLang 需按自身 UB 报告重标定 |

#### 4.2.2 优先级排序的优化项

> 已核实 v0.1.4 未提交 kernel（`src/tileops/_kernels/attention/topk_select_ascend.py`）：直方图（89~163 行）与选索引（185~326 行）都按 `(round==0) or (cached_count>M)` 分支，`cached_count=state[5]` 为上一轮边界桶候选数；≤2048 时走 `cached_key/cached_index` 双缓冲路径、完全不碰 GM，>2048 才退回带前缀过滤的全窗重扫。**逐轮收窄与 UB 复用已在 v0.1.2/v0.1.4 落地**，本规划原第 1、3 项应改写为"缓存溢出退化治理"。

1. **缓存溢出退化治理（修正原第 1/3 项）**。收窄已实现——`cached_key/cached_index` 双缓冲 2048 条、`round%2` 乒乓，第 2 轮起只遍历 `cached_count` 条缓存候选。**未解决的是溢出**：当单桶候选 >2048（全同分/上溢/稠密同桶，坑 11），第 92/189 行退化为"遍历全窗 N + 前缀过滤"，每轮 I/O 回到 O(N)。优化点：溢出时改用 TopKV2 的"GM 只重扫边界桶元素"而非物化全部候选，或对溢出桶做子分桶，避免回到全窗。**前置判断**：先量化溢出在三个主规格 + 128 行诊断输入的频率与延迟占比，频率低就不动——避免过度设计。

2. **Decode 行内并行（大行拆多核）**。确认缺失：Decode `[1,1,16384,1]` 时 R=1、W=`min(1,56)`=1，整行只有 1 个 Vector 核（`T.Kernel(W)` 每 worker 一行，行内仅 `T.SimtVF(threads=H)` 的 H≤256 个 SIMT 线程）。TopKV2 的 `lastDimNeedCore` 把一行沿 KV 轴拆给多核。落地：prefill 保持一致，仅 R 远小于核数时启用行拆分，跨核归并按"有效优先→分数降序→同分取小索引"合并，不能用抢位。

3. **尾轮/单行核利用率**。prefill 尾轮 8 行（32768 % 56）只占 8 核，可与第 2 项合并：尾轮改行拆分铺开。

4. **串行 256 桶扫描用二分替代**。确认存在：第 171 行 `for j in T.serial(256)` 在 `thread==0` 上串行倒序扫 256 桶找 threshold，每轮一次共 5 轮，其余 H-1 个 SIMT 线程空闲；且第 174 行未提前 break，会扫满 256。可用前缀和/二分将 O(256) 串行降为对数级。**前置证据**：坑 8（SIMD 化曾回退）——先量化 256×5 串行步在总延迟占比，再决定是否动。

#### 4.2.3 明确不可迁移（防止下会话再去试）

- **跨核抢位选索引**：TopKV2 多核归并的确定性来自"各自 TopK 后单核复算"（`radix_sort_top_k_inter_core_template_optimization.h`），不是抢位。我们契约要求同分取小索引，任何 `atomic_add` 抢位路径违反坑 12，不采用。
- **原子直方图直接反推选 K**：直方图累加可交换故原子安全，但从原子计数"反推具体选哪些索引"在跨核下无法保证同分顺序；原子只可用于计数，不可用于选索引。
- **MERGE_SORT 大 K 路径**：TopKV2 仅在 `k/n∈[0.25,0.5)` 切排序。主规格 k/n 远低于此，等价"小 K"，纯 radix select 正确；不要在 L=65536、K=1024 下盲目引入全排序，只有窗口极短才重估。
- **Ascend C 参数直接照搬**：`TMP_DATA_NUM=7680` 等是 Ascend C 直接 UB 分配语义，TileLang 抽象不同；移植时以 TileLang 自身 UB/tiling 报告重标定，不能把 7680/1024/32768 当无条件常数。

#### 4.2.4 执行前提

- 任何一项都在解决第 3 节漂移、重建可信同轮对照之后才动手，否则无法区分"优化收益"和"测量漂移"。
- 每项至少做：同进程 AB/BA、固定输入、源码 SHA 一致、原始样本留存，满足第 2.1 节测量纪律。
- 任一主规格回退即标"试验/拒绝"，回退最佳已验证版本，不制造成功结论（沿用第 6.2 节历史文档约定）。

## 5. 有哪些踩过的坑，绝对不要再踩

1. **环境越界。** 只准写 yjl；不能改父目录、公共 Conda、公共 SDK/驱动，也不能重置公共 NPU。仅停止经过 PID、所属用户、cwd 和命令参数核验的本次私人进程，不得按用户名或进程名批量 kill。
2. **错误基线路线。** 旧 ACLNN 路线要求 INT64 indices，需要额外 Cast，不能用于此次 INT32 静态基线。不要使用旧 `cann-topkv2-weekly-baseline-20261029` 的数字冒充正式结果；正确批次是 `cann-topkv2-static-20261030`。
3. **GE 默认降精度。** 必须保留 `must_keep_origin_dtype`，否则 FP32 输出值会发生舍入。不能因为结果“差不多”就接受。
4. **混淆计时。** 不能拿完整 Python 调用事件与纯 kernel 直接算内核加速；不能把 H2D、编译、生成、校验或动态 executor 构造算进基线。profile 与非 profile 独立，差异需要记录。
5. **误信 profiler 退出码。** 曾出现算子表和底层任务表均只有 47/50 条。未补造数据；脚本现为多发 10 次并固定取时间上的最后 N 条，记录实际捕获／发射数量。必须有至少 N 条真实记录，不按延迟挑最快值。历史版本的异常数据仍需保留。
6. **重复聚合旧 trace。** profile 目录必须带 PID／独立运行标识，只解析当前目录。不要把多次重跑 CSV 合并后计算中位数。
7. **忽略跨批次漂移。** 最新控制证明同生成源码当前也明显慢于历史值。此前“UB 一定导致全部回退”的因果判断不成立；不能只修代码而不核查二进制和当前基线。
8. **SIMD 不等于更快。** C=2048 的 SIMD 累计直方图通过 10 项测试但四组均回退；扩大到 16384 曾产生 unsafe ThreadSync hoist/race 警告，不能直接采用。当前工具链的通用 popcount/shfl_sync 未能直接降低，warp_reduce_sum 可用但所试聚合更慢。
9. **计数溢出。** uint16 SIMD histogram 只能用于有界块，跨块必须扩展到 int32。65536 个同分值能直接打穿不正确累计。
10. **缓存不是输入域限制。** 2048 只是候选容量。溢出必须完整计数并重扫；不能只在前 2048 条候选中继续选。
11. **粗筛不保证常数候选。** 同一输入的粗桶候选曾从 4691 降为 835，但全同分、上溢等仍可留下整行。重扫必须同时满足粗桶条件和原始 FP32 前缀条件。
12. **确定性不能靠抢位。** 原子抢位只保证槽位不重复，不保证同分时取小索引。最终同分仍按绝对索引顺序扫描和补足，不用 epsilon 改分数。
13. **Python 嵌套上限。** 在深层 radix body 外再包整轮 early-if 曾触发 `too many statically nested blocks`；别继续叠加深层 if/for。
14. **长时间无输出不代表正常。** 一次完整测试前四项后 CPU 长时间运行，独立用例通过；已保留日志并加单项 180 秒超时。重跑全套通过，不要未经证据把它定性为硬件错误。
15. **Windows 换行。** shell 必须 LF；曾因 `cd ...\r` 失败。remote.py 已对上传 `.sh` 归一 LF。Git 用 input 归一，记录 raw SHA 与 normalized SHA，不能把换行差异误当源代码不可追溯。
16. **临时路径过长。** Python multiprocessing AF_UNIX socket 曾超长，使用 yjl 下短 TMPDIR。CANN cache 目录须先创建。
17. **依赖隔离。** GE 缺少 decorator/attrs/scipy 时装在私人 `build-deps`，没有改全局环境。项目 pyproject 仍 pin TileLang 0.1.12，不要重新 pip install 项目依赖把测试环境 0.1.15 覆盖掉。
18. **硬件数字不能乱推。** 28 Cube 不意味着某个 HBM 带宽；56 blocks 不证明 56 核都在有效计算。没有实测的扩展效率、MFU、资源占用或 Prefill 延迟不能杜撰。

## 6. 写给一个完全没有上下文的新会话

### 6.1 首先执行的核对

本次交接只新增文档提交，不把尚未完成的 v0.1.4 代码一并提交。因此 HEAD 可能多一个 docs 提交；最后的实现标签仍是 `topk-npu-v0.1.3`。先看真实 Git 状态，不要因 HEAD 文档提交误判 v0.1.4 已验收。

```powershell
Set-Location D:\04_Project\TileOPs
git status --short
git log -6 --oneline
git tag --list "topk-npu-*"
```

**不要执行 `git reset --hard`、`git clean` 或直接 checkout 标签覆盖当前文件。** 三个 dirty 文件是未完成候选；先保存 diff 或另建副本再切换。

交接时已额外保存本机备份：`artifacts/topk-optimization-20261030/handoff-v0.1.4-working.patch` 与 `handoff-working-tree.json`。前者是相对实现提交 `1db546f` 的三个候选文件补丁，后者记录原始／归一化 SHA-256；都被 artifacts ignore 排除，换机器时需另行复制。当前工作树已经含有该补丁，不要重复应用。

### 6.2 文件与证据地图

| 路径 | 用途 |
| --- | --- |
| `src/tileops/ascend/__init__.py` | 可选 Ascend backend 注册；源码运行需要显式 import |
| `src/tileops/ascend/topk_select.py` | host adapter、worker/线程/算法选择、输出分配 |
| `src/tileops/_kernels/attention/topk_select_ascend.py` | 当前未提交 v0.1.4 kernel |
| `tests/ops/test_topk_select_ascend.py` | 当前 14 项 NPU 契约／集成测试 |
| `scripts/ascend_topk/measure.py` | 统一预分配原始内核测量；当前有未提交更新 |
| `scripts/ascend_topk/record.py` | 四组证据校验、写版本 JSON；完整记录要求四组源码 SHA 一致 |
| `docs/design/topk-select-optimization-history.md` | 已保存版本和负试验过程 |
| `docs/performance/topk-select-950pr/` | 已提交的 v0.1.0～v0.1.3 紧凑结果 |
| `docs/design/topk-select-ascend-c-implementation-plan.md` | 用户要求参考的 Ascend C 方案，不改变最终 TileLang 交付要求 |
| `docs/design/topk-select-topkv2-performance-analysis.md` | 早期源码与分桶研究，部分假说已被后续试验限制，须结合本交接阅读 |
| `artifacts/cann-topkv2-static-20261030/` | 正式官方静态基线及原始 profiler 数据 |
| `artifacts/topkv2-analysis-20261030/` | 固定公开源码、分桶诊断、初版同输入 profiling |
| `artifacts/topk-optimization-20261030/` | 当前部署／远程控制 helper、逐版本原始证据 |

`artifacts` 被 Git 忽略！**仅 clone Git 不会得到这些 helper 或原始日志**。在同一电脑上继续即可；换机器应另行复制必要 artifacts，同时保留服务器 evidence。旧 `SESSION_STATE.md` 有过期的运行状态，优先以本交接和实际进程为准。

### 6.3 远程操作方法

先向用户获取 SSH 密码，不写进源码、文档、Git 或命令记录文件。以下 helper 都从标准输入读取一次密码；传参方式使用安全的交互输入／获准凭据源，不要把密码写死。

本地 helper：

- `remote.py <stage.sh> [要上传的 artifacts 文件名...]`：上传到固定私人 evidence 后执行该 stage；不要传项目路径来假装部署源码。
- `deploy.py v0.1.4`：仅上传 kernel、host、测试三个白名单文件到私人 checkout；备份旧字节、验证远端字节并写 deployment.json。
- `pull.py`：取回紧凑版本 JSON 与 tests.log，不下载 SDK／缓存。
- `progress.sh`、`process-status.sh`：只读状态；后者仅筛选本次私人目录及指定 pytest 进程。

远端根目录：

```text
/home/f00959234/yjl/tileops-work/evidence/topk-optimization-20261030
```

常用远端命令（必须在通过 helper 或私有 SSH shell 中执行）：

```bash
source /home/f00959234/yjl/tileops-work/evidence/topk-optimization-20261030/base.sh
cd /home/f00959234/yjl/tileops-work/TileOPs
python -m pytest tests/ops/test_topk_select_ascend.py --tileops-device=npu:0 -x -q -s --timeout=180 --timeout-method=thread
```

最终候选测量入口已准备但**尚未执行**：`check-v0.1.4.sh`，先完整测试，后 `measure.py --version v0.1.4 --case all --profile-samples 50`。在解决第 3 节的可信对照问题前，不要直接启动全部大规模跑分。

其他已准备文件：`resource-report.py/.sh` 可从实际选中 kernel 收集编译资源元数据，尚未运行；`control-v2.py/.sh` 已运行并完成；`tune-v4.py/.sh` 已运行完成。不要把“脚本存在”当作“实验完成”。

Ruff 位于 `artifacts/topk-adaptation-20261030/tooling/bin/ruff.exe`。不需要为了格式检查修改全局 Python 环境。

**接手后第一条重要工作：核实当前可信基线与同轮控制，再决定 v0.1.4 是否保留、如何完整测试和提交。不要宣布四轮优化已完成，也不要把历史 CANN 数字和当前异常候选直接拼成达标结论。**
