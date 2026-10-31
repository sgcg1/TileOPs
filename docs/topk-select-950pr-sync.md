# TopKSelect：个人 fork 与 B2Y 分工

2026-10-31 起，代码以个人 fork `https://github.com/sgcg1/TileOPs` 的 `main` 为准。

- TileOPs 仓库：实现、项目测试、设计与正式性能记录。
- B2Y 仓库：`topk-950pr/` 辅助脚本及 Docker 操作指南。
- 服务器：在已有 `yjl` 容器中克隆/更新个人 fork；由操作者从 B2Y 选择辅助文件手动传入。服务器不克隆 B2Y，不通过整项目 ZIP 覆盖源码。

完整操作指南位于 B2Y 的 `topk-950pr/README.md`。默认容器内布局为：

```text
/data/yjl/topk/
  TileOPs/   # git clone 个人 fork，分支 main
  tools/     # 从 B2Y topk-950pr/ 手动传入的文件
  runs/      # 每轮独立原始结果，留在服务器
  tmp/       # 私人临时目录
```

路径、设备号和环境可覆盖，以实际获准配置为准。宿主机执行 `docker exec -it yjl /bin/bash`，随后在容器内执行：

```bash
git clone --branch main https://github.com/sgcg1/TileOPs.git /data/yjl/topk/TileOPs
git -C /data/yjl/topk/TileOPs fetch origin --tags
# 激活已准备好的私人 Python 环境，再加载手动传入的辅助配置。
source /data/yjl/topk/tools/env.sh
bash "$TOPK_TOOLS/run.sh" preflight
bash "$TOPK_TOOLS/run.sh" test
```

已有 clone 时检查工作区干净后用 `git pull --ff-only origin main`，不重复 clone。测试时只暴露指定的一块物理设备，进程内部逻辑设备为 `npu:0`。不自动 pip install 项目，以免 pyproject 的 TileLang pin 覆盖主仓 Ascend 构建。

## 历史整合说明

原本地仓库是独立初始化的快照，分支 `npu/topk-select-950pr` 的保存点为 `e96b1b36`。原来的五个提交、此次候选保存提交及 `topk-npu-v0.1.0`～`v0.1.3` 标签保留，合并将它们接入个人 fork 的历史；没有改写原提交或强推个人仓。

整合以个人 fork 原 main `1301da692d7b1c945f73e94198c5c7052aeeacbc` 为基础，仅迁入 TopK 适配、相关参考验证、测试、本地新增文档和 Ascend backend entry point。上游通用 GEMM、归约、注册模块等保持 fork 内容，不用旧快照覆盖更新。旧快照的完整状态仍可从原分支、标签及合并的历史父提交检索。

辅助测量脚本不放在 main 当前目录，原有脚本仍能从历史提交取回。此前整项目 ZIP 迁移方式已废弃，当前代码同步唯一入口为个人 fork。

## 验证状态

此次为 Git 历史与双仓工作流整合，没有调整候选算法。候选仍未完成新环境的 14 项 NPU 测试和四组正式计时，未发布 v0.1.4 标签。旧服务器的漂移和故障不能推断为新环境结论。

当前 Windows 环境没有 torch、pytest 或 NPU，完成的是 Python 静态语法、Ruff、Bash 解析及辅助汇总的离线数据检查；新容器执行结果需由操作者验证。后续性能必须基于新服务器同轮官方 CANN TopKV2 和匹配输入，不直接引用历史倍数。
