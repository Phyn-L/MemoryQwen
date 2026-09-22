# Token reconstruction + distillation：2026-09-22 best AR 分析

口径：读取本地 .wandb 完整 history，以最大 val_autoregressive/f1 选 step；EM、ROUGE-L 来自同一行。六个新 run 均为 6 GPU、seed 42、batch size 10、4208 steps、11 个 AR 评估点。

| Run ID | T positions | D positions | D weight | Entropy | Best step | AR F1 (%) | EM (%) | ROUGE-L (%) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| fk0ghxmh | 128 | 256 | 0.3 | False | 3200 | 68.437 | 54.590 | 68.202 |
| tptznbow | 128 | 128 | 0.3 | False | 4000 | 67.917 | 53.613 | 67.706 |
| 9u12d028 | 128 | 128 | 0.1 | True | 4000 | 67.779 | 53.711 | 67.510 |
| byw1pzu2 | 256 | 128 | 0.3 | False | 4208 | 67.302 | 52.246 | 67.216 |
| kt7wa3yf | 128 | 128 | 0.1 | False | 3600 | 67.183 | 52.539 | 66.898 |
| lsrfovrj | 128 | 128 | 0.3 | True | 4000 | 66.431 | 52.148 | 66.092 |

本轮最佳 fk0ghxmh：T=1、D=0.3、entropy=false、T positions=128、D positions=256，best F1=68.437。相比本轮 128/128 对照提升 0.520 pp；相比历史 QA-only best 67.974 提升 0.463 pp，但 EM 较低。

D=0.3 时关闭 entropy 提升 1.486 pp；D=0.1 时开启 entropy 提升 0.596 pp，存在条件依赖。Token positions 128→256 下降 0.615 pp。

历史 T+D false best 为 68.290，本轮同配置复跑 67.917；不能将微小增益视为稳定提升。所有结果为验证集单 seed 选峰值，需独立测试集与重复训练确认。

可复算的完整配置与 history：[JSON snapshot](data/loss_token_distill_best_20260922.json)。
