# TopKSelect 950PR 现场核查（2026-10-31）

## 当前结论

已完成本地状态及远端部署、运行库路径和设备状态核查。物理设备 1 当前存在驱动故障告警，暂不执行正式性能验收或继续叠加优化。此告警是本次新观测，**不能据此认定它导致了历史性能漂移**。

## 已核实事实

- 分支为 `npu/topk-select-950pr`，HEAD 为 `fa6ea2b`，最后实现标签为 `topk-npu-v0.1.3`。
- 工作区已有四个修改文件：三个 v0.1.4 候选代码文件及交接文档。交接文档额外包含未提交的 4.2 节规划，全部保留。
- 三个候选代码文件原始 SHA-256 均与 `artifacts/topk-optimization-20261030/handoff-working-tree.json` 一致，Ruff check 通过；没有执行 NPU 契约测试。
- 远端实际导入的 host/kernel 来自私人 checkout，原始 SHA-256 分别为 `c866bef8b87d6cad053a0a32101a12c3919e5e0e99d8bcf9886d99606ff731f4`、`b7a00ac6fff52d12982ef0661974aaea59e98af470b1b4ca149634eb91b55d8b`，与本地一致。
- Python 来自 `yjl/envs/tileops-p1-native`，torch / torch_npu / TileLang 版本为 `2.13.0+cpu` / `2.13.0` / `0.1.15`。主仓上游准确 commit 尚未追溯。
- 已加载 CANN runtime/ACL 库来自私人 9.3.0 工具包；驱动库来自 `/usr/local/Ascend/driver`。`LD_LIBRARY_PATH` 前部包含另一个私人环境 `yjl/envs/tileops/lib`，路径本身不证明发生了库污染。完整映射在原始日志中。
- 设备识别为 `Ascend950PR_9579`，28 Cube / 56 Vector，物理设备选择保持为 1。

## 当前设备异常

北京时间 2026-10-31 10:43:56 开始的只读查询：

- `Health Status: Alarm`。
- `Error Code: 8C4BA00C`。
- 驱动原始信息：`node type=driver_kernel, sensor type=General Software, event state=The software has failed and cannot recover.`
- 当前故障事件数 1，`event id=0x8C4BA00C, severity=2`。
- 同次 usages 查询：CtrlCPU 100%，AI Vector / AI Cube / NPU utilization 均为 0%，HBM 使用率 13%。这些是瞬时读数。
- 初次环境查询还出现设备信息接口错误 `507899`，随后回退；该警告不能单独证明驱动版本不匹配，不据此升级环境。
- 初次 Python 核查曾等待内核锁，随后自行完成，退出码为 0。停止脚本检查时该进程已不存在，没有发送停止信号。尚不能确定等待的具体调用或原因。

未修改共享驱动、SDK、设备时钟或系统配置，未重置设备，未停止其他进程，未启动正式测试或基准。

## 证据与继续条件

本地原始日志（artifacts 被 Git 忽略）：

- `artifacts/topk-optimization-20261030/audit-20261031.sh.log`：设备总览、模块哈希、环境变量和已加载库路径。
- `artifacts/topk-optimization-20261030/health-20261031.sh.log`：健康、故障事件、board 和 usages 查询。
- 同名 shell 脚本已保留在远端私人 evidence 下。

请由服务器管理员处理物理设备 1 的故障，或明确提供另一块可测试设备；不自行切换当前繁忙的设备 0。恢复后先复查设备健康及负载，再进行同轮 AB/BA、实际加载二进制/编译参数核查和现有 CANN 静态模型对照。随后再决定调度候选、执行 14 项测试及四组正式测量。v0.1.4 仍未验收、未提交、未打标签。
