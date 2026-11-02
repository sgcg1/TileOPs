# TopK阶段计时前的最小探针

目的：验证当前主仓TileLang0.1.15/CANN9.3 Weekly上，设备计数器、完成屏障和观察开销是否适合后续阶段诊断。首次现场反馈见下，尚未建立有效阶段耗时。

## 首次现场反馈：源码顺序检查失败

20261102-probe1现场回传status=failed、stage=compile:r1-timed、error=Counter/fence source order changed，脚本哈希与已发布版本一致。已确认Kernel的prelude参数和call_extern接口存在，并已取得r1-timed生成源码供检查；未进入统一50次采样，尚无有效计时校准结果。

当前只收到失败摘要，没有r1-timed.cpp或服务器完整probe-report.json。检查器用全文件字符位置匹配同名标记，假设先定义再调用，没有解析函数作用域；目前不能确认编译器重排或检查器误报。保留断言和原内核，下一步只读导出原失败目录中的源码与标记位置，不重新编译、不运行NPU。

操作步骤已放入B2Y `diagnostics/topk-timer-probe/INSPECT_SOURCE.md`，回传一份带行号的source-inspection文本。收到的各轮结果和此次失败原始摘要按独立ID归档于B2Y `experiments/topk/`，附来源、原文哈希与状态；未回传的信息标明缺失，不覆盖旧数据。

已有H128完整结果见 `topk-select-950pr-scan-20261102.md`。本次不改TopK内核、host、契约测试或现有B2Y topk-950pr工具；探针独立放在B2Y `diagnostics/topk-timer-probe/`，输出与cache单独放在私人runs目录。旧实验和源码身份不受新增目录影响。

## 方案

采用主仓 `T.Kernel(prelude=...)` 和 `T.call_extern` 注入少量计数器/屏障辅助函数。官方文档确认950PR支持AscendC::GetSystemCycle，但当前安装版本的接口和实际降低仍须编译探针确认；接口缺失时保留失败阶段，不盲目更新依赖。

一个Vector worker、H128、4096个FP32值驻留UB。初始值和最终输出拷贝在计数器区间之外。比较同样VF工作量的plain、仅屏障fenced、屏障加计数器timed版本；工作量为1/16/64轮有数据依赖的加1操作，另有0轮timed空区间。输出消费中间状态，逐元素与精确CPU参考比较，减少工作被消除而误测的风险。仍应查看生成源码及必要时的最终指令，源代码存在不证明实际指令区间正确。

每种预热5次，再同进程交替正反序采50次；事件对象、输入、输出及记录缓冲预分配。全部配置做同样的时间戳缓冲回读，回读位于事件区间之外。NPU事件覆盖整个调用，含初始化/写回/调度；counter区间只覆盖中间VF和观察成本。使用64轮减16轮的计数器差/事件差估计单位，避免要求绝对总时间相等。

CANN对950的文档值为1000 ticks/μs，与Aicore curFreq字段不同。10%偏差窗口是诊断启发式，不是认证标准；不符合时不自动换倍率。记录屏障和计数器造成的事件百分比变化与空区间原值，不机械扣除开销，也不把负变化裁成0。

## 成功标准及限制

先确认编译完成、输出正确、时间戳确实被写入且递增、工作量变化与counter区间相符，再看尺度检查和观察扰动。探针的 `stage_profiling_accepted` 始终为false；completed只指采集完成。工具通过不证明TopK每个VF、原子统计或同步区间都能同样计时。

只有通过审阅，才扩展为TopK每轮histogram、scan、compact、状态更新、tie阶段记录，并同时保存候选数量、溢出重扫信息。多worker的阶段时间不能直接相加当作墙钟耗时。额外屏障可能改变原流水，需要与无插桩版总耗时比较；仿真周期与设备耗时分开报告。

本地验证范围为Python/Ruff、11项CPU解释与只读源码导出测试、文档Bash语法和文件哈希。本地未运行NPU；现场首次回传及其限制见上。当前先按INSPECT_SOURCE.md回传源码，计时方法仍未通过验证。

参考：[GetSystemCycle的950换算与同步示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)、[主仓Kernel prelude](https://github.com/tile-ai/tilelang/blob/main/tilelang/ascend/language/kernel.py)、[主仓诊断流程](https://github.com/tile-ai/tilelang/blob/main/.agents/skills/tilelang-ascend/references/profiling.md)。
