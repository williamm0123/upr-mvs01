# 第一轮 (2026-08-20) 的 arm 脚本 —— 已作废, 不要提交

作废的两个原因, 都不是配置本身的问题:

1. **fp16 溢出**: 八个 arm 里五个死于 stage4 前向非有限值 (只有 stage4 的 loss
   分量坏、输入全部有限、`depth_full` 的 finite_frac 恰好 0.5)。第二轮统一改用
   `--amp-dtype bf16`。

2. **LR horizon 与停止步数没解耦**: `--steps 12000` 时余弦在 12k 就退火到底,
   所以这些 arm 在 8k-12k 是"已退火的模型" (lr 1.6e-5), 而 30k 历史 run 在同一
   步数窗口是 2.3e-4 —— 差一个数量级。既不能跟历史横比, 选中的 checkpoint 也
   不能无跳变续训到 30k。第二轮统一 `--lr-schedule-steps 30000`。

留着是为了对照当时跑出来的数据 (log/experiments/_archive 与 tensorboard)。
第二轮的 arm 在 scripts/arm_R32_*.sh。
