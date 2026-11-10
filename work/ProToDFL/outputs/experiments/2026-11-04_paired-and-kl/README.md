# 2026-11-04 配对聚合检查与 KL 收益

本期主题：在保存的快照上做「同一起点、同一批邻居、只换聚合范围」的配对检查，以及 KL 与单伙伴聚合收益的关系。两项都不训练模型。

包含：

| 目录 | 内容 | 脚本 |
|---|---|---|
| `checks/paired_agg_20c/` | 20 客户端、5 epoch 快照的配对聚合检查（聚合前 / 全参数 / 只聚合 base 的准确率与损失） | `scripts/paired_aggregation_check.py` |
| `checks/paired_agg_5c/` | 5 客户端小规模对照 | 同上 |
| `checks/kl_benefit_20c/` | 1 epoch 快照上的 KL 分组与按对的收益表（`kl_benefit_clean.csv`） | `scripts/kl_benefit_distribution.py` |
| `checks/kl_benefit_5c/` | 5 客户端小规模对照 | 同上 |
| `checks/kl_benefit_server/` | 5 epoch、`srv20-e5-base-s0/round_1` 上的 380 对收益表（`kl_benefit.csv`），后续同目标分析的主输入 | 同上 |

对应记录：`docs/02_实验记录.md` 中 2026-11-04 的条目（RUN-012～RUN-019）；KL 部分的结论边界见 `docs/09` 第 4、9.5 节。

结论以 `docs/02` 与 `docs/09` 为准，本 README 不重复数字。
