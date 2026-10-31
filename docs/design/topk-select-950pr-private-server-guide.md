# 在保密服务器继续 TopKSelect 950PR 工作

## 1. 目标与边界

本指南对应 2026-10-31 的工作区候选。v0.1.4 尚未验收，不能将迁移包叫作正式 v0.1.4 发布。最终实现使用 TileLang 主仓 Ascend 后端，对照使用官方 CANN TopKV2。先建立新服务器同轮基线，再选调度，最后做完整验收。

假设允许把本地源码通过 B2Y/GitCode 导入新环境。服务器上的代码、结果、环境信息是否可导出，由当地保密规定决定；本指南不要求上传保密数据。GitCode 只用作你操作的文件传输渠道，不要求把 B2Y 仓库替换成 TileOPs。

新机器的账号、目录、设备号、CANN 安装、Python 和 TileLang 构建情况均未知，以下变量必须按现场填写。没有可用的主仓 Ascend TileLang 安装时，应先由环境负责人完成构建配置；单独拷贝本算子源码不能替代编译运行环境，也不要直接安装独立 tilelang-ascend 仓。

## 2. 本地经 B2Y 传输

本地迁移包：`artifacts/topk-migration-20261031/topk-private-server-kit.zip`。

包中包含：

- `TileOPs/`：当前工作区的 src、测试、workloads、必要配置及测量脚本，包括三个未提交候选文件。不是仅从 HEAD 导出。
- `support/kernel-v0.1.0.py`、`support/kernel-v0.1.2.py`：从已保存标签提取的对照 kernel。
- `support/cann/`：官方 TopKV2 的 GE 静态模型构建入口、协议头和测量脚本；不含旧服务器 SDK、缓存、模型二进制、SSH helper 或密码。
- `MANIFEST.json`：包内文件 SHA-256。

包内 `measure.py` 和 CANN `benchmark.py` 仅把固定物理设备号 1 的元数据改成读取 `ASCEND_RT_VISIBLE_DEVICES`；源代码换行归一为 LF。它们与原工作区字节不完全相同，正式测量以包内 manifest 和新结果哈希为准。

在你自己的 B2Y checkout 建立独立目录，例如 `topk-transfer/20261031/`，复制 ZIP，先查看该仓库状态，再只提交 ZIP 和你需要的说明文件：

```bash
git status --short
git add topk-transfer/20261031/topk-private-server-kit.zip
git diff --cached --stat
git commit -m "Add TopK private-server migration kit"
# 使用 B2Y 既定分支和 GitCode 远端流程自行 push。
```

不要执行 `git add .` 来顺带上传 B2Y 中其他文件。此处没有替你推送或修改任何远端。

## 3. 在新服务器解包、校验

先通过组织允许的方式更新 B2Y，再选择你获准写入的私人目录。下列命令在 Linux bash 执行：

```bash
export TOPK_ROOT=/ABSOLUTE/AUTHORIZED/PATH/topk-work
export B2Y_ROOT=/ABSOLUTE/PATH/TO/B2Y
mkdir -p "$TOPK_ROOT/kit"
python -m zipfile -e "$B2Y_ROOT/topk-transfer/20261031/topk-private-server-kit.zip" "$TOPK_ROOT/kit"
export TOPK_REPO="$TOPK_ROOT/kit/TileOPs"
export TOPK_SUPPORT="$TOPK_ROOT/kit/support"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['TOPK_ROOT']) / 'kit'
for name, expected in json.loads((root/'MANIFEST.json').read_text()).items():
    if name.startswith('_'): continue
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('manifest OK')
PY
```

只向新建目录解包；不要覆盖已有工作。反复迁移时换目录，避免混入上一次结果。

## 4. 激活已有环境并隔离输出

先按现场提供的方法加载 CANN，再激活已准备好的 Python 环境。以下两行是示意，必须换成实际路径：

```bash
source /ACTUAL/CANN/set_env.sh
conda activate /ACTUAL/PRIVATE/PYTHON/ENV
```

不要执行 `pip install -e .` 或让项目依赖解析器自动安装依赖：pyproject 将 TileLang 固定为 0.1.12，可能覆盖已准备好的 Ascend 构建。通过 PYTHONPATH 运行源码即可。缺 pytest、pytest-timeout、PyYAML 等时，只在获准的私人环境使用内部包源补齐；不要安装整个 CUDA bench/dev 依赖组。

```bash
export TOPK_DEVICE=0  # 改为现场分配的一个物理设备号，不照抄旧服务器的 1
export ASCEND_RT_VISIBLE_DEVICES="$TOPK_DEVICE"
export RUN_ID="newhost-$(date +%Y%m%d-%H%M%S)"
export EVIDENCE="$TOPK_ROOT/runs/$RUN_ID"
mkdir -p "$EVIDENCE" "$TOPK_ROOT/tmp" "$EVIDENCE/cache/cann" "$EVIDENCE/cache/work" "$EVIDENCE/logs"
export TMPDIR="$TOPK_ROOT/tmp"
export XDG_CACHE_HOME="$EVIDENCE/cache"
export TILELANG_CACHE_DIR="$EVIDENCE/cache/tilelang"
export ASCEND_CACHE_PATH="$EVIDENCE/cache/cann"
export ASCEND_WORK_PATH="$EVIDENCE/cache/work"
export ASCEND_PROCESS_LOG_PATH="$EVIDENCE/logs"
export TORCH_EXTENSIONS_DIR="$EVIDENCE/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$EVIDENCE/cache/inductor"
export PYTHONDONTWRITEBYTECODE=1
export TASK_QUEUE_ENABLE=0
export PYTHONPATH="$TOPK_REPO:$TOPK_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0
set -o pipefail
cd "$TOPK_REPO"
```

只暴露一个物理设备后，程序内部使用逻辑 `npu:0`。不要把 pytest 的设备参数改成物理号。不要重定向 HOME；TMPDIR 保持短路径。

这些变量应保存在新服务器私人 `activate-topk.sh` 中；新终端先加载它。每次新实验给 EVIDENCE 新目录，避免旧 profiler 聚合和 JSON 覆盖。

## 5. 现场检查：通过才继续

```bash
npu-smi info | tee "$EVIDENCE/device-before.txt"
npu-smi info -t health -i "$TOPK_DEVICE" | tee "$EVIDENCE/health-before.txt"
npu-smi info -t current-fault-event -i "$TOPK_DEVICE" | tee "$EVIDENCE/fault-before.txt"
python -u - <<'PY' | tee "$EVIDENCE/environment.txt"
import importlib, os, sys
import torch
import torch_npu
import tilelang
print('python', sys.executable)
print('versions', torch.__version__, torch_npu.__version__, tilelang.__version__)
for name in ['tilelang', 'tilelang.ascend.language', 'tilelang.ascend.backend',
             'tileops.ascend.topk_select', 'tileops._kernels.attention.topk_select_ascend']:
    print(name, importlib.import_module(name).__file__)
assert torch.npu.is_available()
torch.npu.set_device(0)
print(torch.npu.get_device_name(0))
print(torch.npu.get_device_properties(0))
print('memory', torch.npu.mem_get_info())
print('physical', os.environ['ASCEND_RT_VISIBLE_DEVICES'])
print('CANN', os.environ.get('ASCEND_HOME_PATH'))
print('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH'))
PY
```

通过条件：设备健康正常、确实是目标 950PR、模块来自预期主仓 Ascend 安装和解包源码、具备 `vector_core_num`。主 Prefill 分数输入约 8 GiB，额外还需输出、校验临时张量和运行库空间；以脚本内存检查及实际空闲内存为准。设备繁忙时不要做正式性能判断。

如果设备告警、初始化超时或版本接口缺失，先定位环境；不要靠修改共享驱动、重置 NPU 或更换算子精度绕过。旧服务器故障不是新服务器必然存在的问题。

## 6. 正确性测试

先确认收集到 14 项，再执行；不要加 xdist 并行跑：

```bash
mkdir -p "$EVIDENCE/candidate"
python -m pytest tests/ops/test_topk_select_ascend.py \
  --tileops-device=npu:0 --collect-only -q 2>&1 | tee "$EVIDENCE/collection.log"
python -u -m pytest tests/ops/test_topk_select_ascend.py \
  --tileops-device=npu:0 -x -q -s --timeout=180 --timeout-method=thread \
  2>&1 | tee "$EVIDENCE/candidate/tests.log"
```

必须看到 **14 passed**。全部 skipped 不算通过。失败时停止性能验收，保留第一个失败用例和 traceback，在该用例上复现，不能删除特殊值测试。

契约包括 FP32 原值选择、NaN/Inf/±0、FP16 边界与溢出、全同分、缓存溢出、K>2048、窗口、分组、非连续输入和输出布局。`torch.compile` 只覆盖 eager backend/fullgraph，不代表 Inductor 或 NPU graph 已验证。

## 7. 同进程控制实验

迁移包新增 `control_portable.py`，在同一进程比较初版、v0.1.2、当前候选 H128/H256；固定输入、预分配缓冲、正反顺序交替，每个版本 50 次事件。它不会覆盖当前 kernel 文件。这里只测有限随机输入，不能替代第 6 步。

```bash
timeout --signal=TERM --kill-after=10s 600s python -u scripts/ascend_topk/control_portable.py \
  --support "$TOPK_SUPPORT" --output "$EVIDENCE/control-decode-1" \
  2>&1 | tee "$EVIDENCE/control-decode-1.log"
timeout --signal=TERM --kill-after=10s 600s python -u scripts/ascend_topk/control_portable.py \
  --support "$TOPK_SUPPORT" --output "$EVIDENCE/control-decode-2" \
  2>&1 | tee "$EVIDENCE/control-decode-2.log"
```

结果在各目录 `control.json`，同时保存生成的 C++。输出目录必须是新目录。v0.1.2 与当前 H128 的生成代码如果相同，事件耗时应在运行波动范围内接近；异常则继续查实际二进制、编译选项、加载库、设备负载，暂不选“优化赢家”。初版与其他版本代码不同是正常的。

测 Prefill workers 时先读取实际 Vector 核数，仅对可用核数做小规格诊断，例如机器确有至少 56 个 Vector 核时：

```bash
python -u scripts/ascend_topk/measure.py --version tune-w32 --case small --workers 32 --threads 128 --profile-samples 50
python -u scripts/ascend_topk/measure.py --version tune-w56 --case small --workers 56 --threads 128 --profile-samples 50
python -u scripts/ascend_topk/measure.py --version tune-w56-repeat --case small --workers 56 --threads 128 --profile-samples 50
python -u scripts/ascend_topk/measure.py --version tune-w32-repeat --case small --workers 32 --threads 128 --profile-samples 50
```

这种 W32/W56 比较是相近时段正反顺序的独立进程诊断，不能称作同进程 AB/BA。控制脚本可以用 `--rows 128 --length 65536 --k 2048 --workers 32` 在固定 W 下比较版本，但不会跨 W 同进程比较。这里只用于筛选，完整规格仍要验证。

## 8. 在新服务器重建官方 CANN 基线

不要拿旧服务器的 8.721 μs / 50.65 ms / 52.08 ms 直接算新服务器的性能倍数。`.om` 在新环境重建，不复制旧编译缓存。

```bash
mkdir -p "$EVIDENCE/cann"
cp "$TOPK_SUPPORT/cann/"* "$EVIDENCE/cann/"
cd "$EVIDENCE/cann"
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 build.cpp \
  -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/include/external" \
  -I"$ASCEND_HOME_PATH/include/external/graph" -I. \
  -L"$ASCEND_HOME_PATH/lib64" -lge_compiler -lgraph -lgraph_base -lge_runner \
  -o build 2>&1 | tee compile.log
./build 1 16384 2048 decode 2>&1 | tee build-decode.log
./build 32768 65536 1024 prefill-k1024 2>&1 | tee build-prefill-k1024.log
./build 32768 65536 2048 prefill-k2048 2>&1 | tee build-prefill-k2048.log
```

这个构建入口针对 `Ascend950PR_9579`，ABI=0 和链接库列表来自旧环境；头文件/ABI 不同须在私人构建目录按现场 SDK 调整并记录，不更改 SDK 本身。若芯片不是该型号，应先确认任务适用性再改 SOC。GE 编译如果缺 decorator/attrs/scipy，通过内部依赖源补在私人环境；缺编译器或 GE 库则找环境负责人。

必须保留 `must_keep_origin_dtype`，以及 largest=true、sorted=false、INT32 indices、sort_policy=0、dim=-1。模型只调用官方 TopKV2，不自行换成 AscendC::TopK 封装或其他基线。

```bash
EVIDENCE="$EVIDENCE/cann" python -u benchmark.py 1 16384 2048 decode 2>&1 | tee decode.log
EVIDENCE="$EVIDENCE/cann" python -u benchmark.py 32768 65536 1024 prefill-k1024 2>&1 | tee prefill-k1024.log
EVIDENCE="$EVIDENCE/cann" python -u benchmark.py 32768 65536 2048 prefill-k2048 2>&1 | tee prefill-k2048.log
python profile-summary.py | tee profile-summary.txt
cd "$TOPK_REPO"
```

每条前置 EVIDENCE 只影响那次 Python 子进程，不改变外层实验根目录。先 Decode 成功再执行 Prefill，逐项检查退出码。CANN 脚本采集 50 次事件和独立 50 次 profile；`profile-summary.txt` 必须确认每组真实 TopKV2 记录达到 50，不能只有 47 条仍宣布有效。旧 baseline 脚本没有额外启动余量：不足时在新的实验目录重跑，不合并旧目录。若需改为多发 10 次，则必须同时实现按设备时间取最后 50 条并记录规则，不能只改循环数量。

查看 task_time.csv，确认没有额外 Cast、Transpose、AICPU 计算任务；模型控制任务允许存在。每个正式基线目录只跑一次，失败重跑用新目录，避免 profile-summary 递归聚合旧 trace。

## 9. 四组统一测量及验收

回到源码目录，在确认计时可信后，测当前默认配置：

```bash
cd "$TOPK_REPO"
python -u scripts/ascend_topk/measure.py --version candidate --case all \
  --samples 50 --profile-samples 50 2>&1 | tee "$EVIDENCE/candidate/benchmark.log"
```

四组为 Decode `[1,1,16384,1],K2048`，诊断 `[1,128,65536,1],K2048`，两个完整 Prefill `[1,32768,65536,1],K1024/K2048`。

当前 host 默认 H128，workers=min(rows,实际 Vector 核数)，L65536/K1024 走直接 FP32 radix，其余粗筛。命令行线程/worker 覆盖只用于实验，不会自动修改 host 默认。若选定 H256 等新默认，需要先改 host，再重跑全部 14 项和四组测量，最后记录新源码哈希。

当前测量器有固定随机输入哈希断言。新 torch/RNG 栈若生成不同输入，断言失败时先停止；不要随手删断言。可选择在新服务器生成一份私人 FP32 输入，修改两个测量入口都读取相同输入，再按实际 SHA 建立独立新基线并记录改动。数据不必导出，旧哈希也不再作为新批次预期值。

结果有效条件：每组 50 次事件、50 条真实 profile；kernel 源码哈希四组一致；同规格 CANN/TileLang 输入哈希相同；采样中无编译、H2D、随机数和校验。代码模板的生成 C++ 会随 shape/K 改变，四组生成 C++ 哈希不要求相同。profile 与事件独立记录，不互相替换。

性能倍数 = 同轮 CANN 纯内核中位数 / TileLang 纯内核中位数。事件对事件可单列，但静态模型事件包含模型调度，不能标成纯内核。128 行无官方同规格基线，不换算“×Ascend C”。不以 MFU/95% 宣称达标。

`record.py` 是旧版本记录入口，仅校验 profile >=10，且会生成版本标签字段；新机器首次迁移不要直接用它把候选标成正式 v0.1.4。先按上述更严格 50/50 规则检查并保留原始结果。只有代码/环境/输入、14 项和全部主规格结论明确后才整理正式版本；任一主规格回退就记 experimental/rejected，不制造整体收益。

## 10. 保存、回传与下一轮协作

服务器本地至少保留：包内 manifest、源码修改 diff、环境版本、测试日志、每组 JSON、生成 C++、当前实际编译参数及二进制哈希、原始 profiler CSV、设备测前测后状态。实际加载二进制如何定位取决于现场 TileLang 构建，控制脚本目前只保存生成源码，不能声称已经核对二进制一致。

允许出域时，可在 B2Y 的专用新目录放经审核的紧凑汇总，不必上传全部日志。推荐字段：

```text
实验编号：使用无业务含义的代号
环境：是否目标 950PR；软件版本是否可公开
正确性：14 passed / 失败用例；不是 skipped
控制：各版本 H/W、50 次事件中位数、重复波动、输入是否相同
主规格：shape、K、输入是否匹配、CANN/TileLang 内核中位数（统一 μs）
采样：event=50、profile=50、计时范围、是否存在额外计算任务
实现：包内源码是否修改；可公开的 diff 或摘要
结论：保留/拒绝/证据不足；下一项问题
```

原始 profiler、环境映射、异常栈可能包含路径、用户名、主机信息；是否能导出由当地审核决定。若不能传任何结果，你仍可依照本文在服务器内部完成判断；这里仅讨论抽象问题。上传前只显式 git add 审核过的文件，不上传 SDK、缓存、`.om`、凭据或业务输入。若允许传代码 diff，我可以据此准备下一轮修改，再经 B2Y 由你导入执行。

遇到失败，回报允许披露的“阶段编号 + 首个错误类型 + 退出码 + 是否修改源码”即可；不要求为了诊断导出完整保密日志。
