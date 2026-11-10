# 2026-11-02 启动检查与 hop 语义

本期主题：环境启动是否正常、`hop` 字段到底是什么、18 客户端下启动是否稳定。不训练新模型，全部是检查类实验。

包含：

| 目录 | 内容 | 入口 |
|---|---|---|
| `checks/startup-seed0/` | 20 客户端启动检查（首轮能否跑通、日志与通信统计是否落盘） | `scripts/run_smoke.ps1` |
| `checks/hop-semantics/` | 核对 `hop` 列在图上是邻居跳数还是其他含义 | `scripts/` 下的核对脚本，结果见 `report.md` |
| `checks/startup-seed0-18c/` | 18 客户端启动检查 | 同上 |

对应记录：`docs/02_实验记录.md` 中 2026-11-02 的条目；日志在 `outputs/logs/cifar10/`，按 tag 名对应。

结论以 `docs/02` 与 `docs/09` 为准，本 README 不重复数字。
