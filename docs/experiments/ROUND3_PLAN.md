# 第三轮运行计划（2026-09-17 夜 → 次日）

> 历史快照：2026-09-18 归档。正文状态、数值、路径和预计完成时间属于当时记录，本次未重新运行实验或核实远端状态。当前操作见[运行指南](../guides/RUNNING.md)，实验状态见[实验索引](../experiments/README.md)。

代码状态：两台机器都在 `718b606`（5 项修复 + 机器表 + 单一评测脚本）。下面所有数字都出自
**同一份修复后代码**，可以直接并到一张表里。第二轮的完整结果记在
[`docs/experiments/AB_H200.md`](AB_H200.md) 第 10 节。

## 0. 现在的状态

* **第二轮就是第一对"跑完的、只差 6 个开关"的 A/B**（同形状 ctx1024/M64 = 16:1、同 global 64、
  同 1 epoch，TF 200 / AR 400 / save 400 / warmup 200）。同 step 3600 对齐：

  | 指标 @3600 | ON | OFF | Δ |
  | --- | --- | --- | --- |
  | AR f1 / em | **0.6730 / 0.5352** | 0.5716 / 0.4297 | +0.101 / +0.106 |
  | AR first_token_em | **0.6934** | 0.6250 | +0.068 |
  | TF f1 / em | **0.6862 / 0.4118** | 0.6051 / 0.3476 | +0.081 / +0.064 |
  | TF ppl | **2.156** | 2.656 | −0.500 |
  | memory 探针（nats/token） | **7.279** | 7.402 | −0.123 |

  ON 终点（3945）：AR f1 0.6737 / em 0.5342 / rouge_l 0.6715 / first_token_em 0.6953；
  TF f1 0.6882 / em 0.4121 / ppl 2.151 / 探针 7.284；`train/lr` 单条 cosine 1.25e-5 → 7e-9。
* **ON 已经在 1 个 epoch 内收敛**：最后 1,145 步（2800 → 3945）TF f1 0.6838 → 0.6882、
  ppl 2.186 → 2.151、探针 7.312 → 7.284，AR f1 只再涨 +0.02。
* OFF 臂当时读到 step 3600（AR f1 0.5716），还差约 145 步 + 终点评测 → 预计 **00:25 前后结束**，
  终点落在 AR f1 ≈ 0.585 ± 0.01，所以 ON−OFF ≈ **+0.09 AR f1 / +0.09 em**（幅度远超
  AB_H200.md 第 5.4 条 0.01-0.02 的判读门槛）。

**两个推论**：① "reader 开关在同形状下有效"可以当结论用了；② **训练量不是下一个瓶颈**
（再训 epoch 的上限看起来只有 +0.02），下一轮该动的是**形状 / 压缩比**。

## 1. 今晚（约 1 h，不需要训练）

1. 等 OFF 收尾，取两臂终点：
   ```bash
   cd /home/lijie/proj2/xmu/lz/MemoryQwen
   python extract_run.py wandb/offline-run-20260917_230502-y3kst0an/run-*.wandb | tail -25
   ```
2. **全集 SQuAD v1/v2 评测**（4 卡；v1 10570 行 / v2 5928 / v2all 11873；各约 10-15 min）。
   先加 `SAMPLE_CAP=64` 各过一遍确认路径与显存，再放开：
   ```bash
   cd /home/lijie/proj2/xmu/lz/MemoryQwen
   CHECKPOINT=outputs/ab_h200_on/Qwen1.7B_20260917_213648/last.pt \
     NUM_PROCESSES=4 WORK=outputs/eval_on  bash scripts/archive/eval_squad_v1v2.sh
   CHECKPOINT=outputs/ab_h200_off/Qwen1.7B_20260917_230444/last.pt \
     NUM_PROCESSES=4 WORK=outputs/eval_off bash scripts/archive/eval_squad_v1v2.sh
   ```
   产物 `outputs/eval_{on,off}/results.md`：AR + TF 的 EM/F1/ROUGE_L，分 v1.1 / v2.0 / v2.0-all。
   （ON 的 checkpoint 在 4090 上也有一份，评测在哪台机器跑都行；OFF 目前只在 H200。）

## 2. 第三轮主线：压缩比轴（ON 固定，单变量，约 2.1 h）

第二轮固定了 ctx=1024，只把 M 从 16 提到 64。现在沿同一根轴继续压：**只改 `memory_length`**，
其余（含 6 个开关、global 64、cadence）逐字段与第二轮 ON 相同 —— 与它的终点构成单变量对照。
两个配置已经写好，diff 只有 `memory_length` + `output_dir`：

| 跑 | 配置 | 压缩比 | 预期墙钟 |
| --- | --- | --- | --- |
| R3-1 | `configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml` | 32:1 | ~1.1 h |
| R3-2 | `configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m16.yaml` | 64:1 | ~1.0 h |

```bash
cd /home/lijie/proj2/xmu/lz/MemoryQwen
export PATH=/home/lijie/proj2/.conda/envs/shine/bin:$PATH   # python 与 accelerate 必须同环境
CONFIG=configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml NUM_PROCESSES=8 bash scripts/train.sh
# 跑完接着第二条（也可以在另一个 tmux 窗口里排队）：
CONFIG=configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m16.yaml NUM_PROCESSES=8 bash scripts/train.sh
```

跑起来先看两行：`schedule: machine=h200 steps=3945 batch=8x8 ranks=8 ...`（不对就停），
以及第一个 `val_teacher_forced/*`（step 200）。跑完同样用 `scripts/archive/eval_squad_v1v2.sh`
把两个 checkpoint 在全集上评一遍，这样压缩比曲线和 ON/OFF 用同一把尺子。

**判读（对照点 = 第二轮 ON：M64 0.6737 AR f1 / 0.6882 TF f1 / 探针 7.284）**：

* M16 仍然 ≥ 0.65 AR f1 → "1,024 token 的上下文压成 16 个 memory token 几乎不掉点，靠的是开关"
  —— 这是方法最值钱的一句话，可以直接进论文；
* M16 落在 0.58-0.65 → 操作点取 M32，把 M16 当成"压缩极限"的边界点；
* M16 < 0.55 → 用 `first_token_em`（检索）与 `reconstruction_loss`（memory 容量）定位是检索先崩
  还是容量先崩，再决定是调 `readout_length` / `distill_weight` 还是回到 M32。

**R3-3（条件跑）**：如果 M16 站住了，补一臂 **OFF @ M16**（`ab_h200_on_m16.yaml` 把 6 个开关翻回
OFF 值即可，~1.0 h）→ 得到"开关收益随压缩比变化"的交互图。预期差距随压缩比拉大；若反而缩小，
说明开关只在宽 memory 下有用，那是另一个结论。

## 3. 次选：上下文长度轴（R3-1 之后，约 2.5 h）

`configs/4090/qwen-1.7b/context_length/train_reader-on_ctx2048_m64.yaml`（ctx2048 / M64 = **32:1**，batch 4 × 8 = global 32，
1 epoch = 7,974 步，2.2-2.9 h）早就写好了，一直没跑。它和 R3-1（ctx1024/M32 = 32:1）在**同一压缩比**
下只差上下文长度：两条都跑完，才能把"比值"和"长度"分开——否则第三轮结束时仍然不知道
第二轮 OFF 大形状（ctx2048/M64）赢在小形状（ctx1024/M16）上，是因为上下文更长还是因为压缩更浅。

```bash
CONFIG=configs/4090/qwen-1.7b/context_length/train_reader-on_ctx2048_m64.yaml NUM_PROCESSES=8 bash scripts/train.sh
```

（它 batch 4、global 32、warmup 400、save 800；和 global 64 的那几条按"看过的 context 数"对齐比较，
不要按 step 对齐。）

## 4. 现在不建议做

* **3 个 epoch**：ON 的 TF 侧在一个 epoch 内就平了，AR 上限 +0.02 —— 性价比远低于压缩比。
* **单开关消融**（A1 tied 头 / C3 resampler 各自贡献）：等形状曲线定下来再拆；在错的形状上做消融
  要重跑。
* **resume 旧代码/旧形状的 checkpoint**：口径不同，跑出来的数不能和新的一起比。
* **数据混合实验**（去掉占训练量 74% 的 ms_marco）：便宜，但它回答的是"SQuAD 绝对水平"，
  不影响"开关/压缩比"这两个主线问题，排在后面。

## 5. 每次跑前/后的固定动作

* 跑前 `DRYRUN=1`（A/B 用 `MACHINE=h200 DRYRUN=1 NUM_PROCESSES=8 bash scripts/archive/run_ab.sh`；
  单跑直接看上面命令里的 `CONFIG=`/`NUM_PROCESSES=`），确认 `schedule:` 行里的
  `steps=`、`batch=…x… ranks=…` 和预期一致。
* 跑起来先看 `train/lr` 是不是**单条** cosine（多卡重复缩短过的问题已修，见 `0f444c2`），
  以及第一个 TF 评测是否正常。
* OOM / 断点之后：`RESUME=1`（A/B 脚本）或 `bash scripts/train.sh --resume <ckpt>`，
  **rank 数必须与失败那次一致**。
* 跨 run 比较前确认三件事：同一份代码、同 global batch、同评测口径（`validation_max_samples`
  数 context，`autoregressive_max_qa` 是**全局**行数）。

## 6. 磁盘（不急）

`/mnt/shared-storage-user/evoagi-share/lijie` 还有 **2.6 T** 空闲（`outputs/` 只占 82 GB），
所以清理纯属顺手。真要腾地方：`rm -f outputs/ab_h200_off/*/step-*.pt`（保留 `last.pt`，
省 ~18 GB）、`rm -rf outputs/Qwen1.7B_20260917_012511`（21 GB，旧代码 run，只作历史）。
**两臂的 `last.pt` 不要删** —— 第 1 节的评测要用。
