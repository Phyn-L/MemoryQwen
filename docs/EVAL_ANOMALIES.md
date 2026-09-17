# 全集 SQuAD v1/v2 评测（ON vs OFF）异常核查

对象：`outputs/eval_{on,off}/results.{md,json}` + `test_{v1,v2,v2all}.log`，2026-09-18 00:20-00:38，
checkpoint 是第二轮那对（`ab_h200_on/Qwen1.7B_20260917_213648/last.pt`、
`ab_h200_off/Qwen1.7B_20260917_230444/last.pt`）。六份日志里没有报错、没有 NaN。

## 0. 先看没问题的部分

| 子集 | ON AR em / f1 | OFF AR em / f1 | Δ f1 | ON TF f1 / ppl | OFF TF f1 / ppl | ON 探针 | OFF 探针 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1.1（10570 行） | **0.5150 / 0.6445** | 0.4108 / 0.5448 | +0.0997 | **0.6842** / 2.168 | 0.5844 / 2.791 | **7.289** | 7.402 |
| v2.0 answerable（5928 行） | **0.5068 / 0.6408** | 0.4050 / 0.5384 | +0.1025 | **0.6821** / 2.198 | 0.5830 / 2.853 | **7.259** | 7.397 |

* ON 在每个子集、每个指标上都领先，幅度与训练期曲线一致（AR f1 +0.10、TF f1 +0.10、
  qa_loss 差 0.21-0.26 nats、探针差 0.11-0.14）→ **第二轮的 A/B 结论在全集上复现**。
* v2 与 v2all 的 AR/TF 四个指标**逐位相同**（`...4328892 / ...639436 / ...3913 / ...1712`），
  但 `reconstruction_loss` 差 0.002-0.006 —— 这正好说明指标是逐行均值（对行序不敏感），
  而 loss 受 batch 组成影响。这条"同值"是刻意的自检，也是下面第 1 条异常的证据。

## 1. 【真 bug】`v2.0 (all)` 这一列是假的：无答案行在数据层就被丢掉了

证据链：

1. `outputs/eval_on/config_v2all.yaml`：`filter_no_qa: false`、`root=.../v2all`（脚本按设计生成的）。
2. `counts_v2all.json` = `{"contexts": 1204, "pairs": 11873}` → 磁盘上的 split 文件确实是 11,873 行
   （5,649,779 B，其中 5,945 行是 unanswerable）。
3. 但数据集缓存的 metadata（`outputs/Qwen1.7B/dataset_cache/validation-75839f4b21c4c3e3/records.meta.json`）
   是 `contexts=1204, qa_pairs=5928, filter_no_qa=False, size=5649779` → **建库时把 5,945 行丢了**。
4. 于是 v2all 的评测集与 v2 完全相同（5928 行），指标逐位一致。

代码位置（`src/data.py::_build_records`）：

```python
for qa in row.get("qa_pairs", []) or []:
    answers = _answers(qa); question = str(qa.get("question", "")).strip()
    if filter_no_qa and (not question or not answers):     # 125: 开关只管"无 question / 无 answers"
        continue
    if question and answers:                               # 127: 无条件要求有答案 ← bug
        pairs.append(QARecord(question, answers[0], ...))
if pairs:                                                  # 131: 只剩无答案行的 context 也被丢
    records.append(...)
else:
    dropped_no_qa += 1
```

`filter_no_qa` 的本意是"是否保留没有任何可用 QA 的 **context**"，但第 127 行让任何**无答案的 QA 行**
都进不来，第 131 行再把"只剩无答案行"的 context 整条丢掉 —— 所以 `filter_no_qa=False` 拿不到
unanswerable 行，`results.md` 里那列的表头 `v2.0 (all; unanswerable scored as misses)` 和
`subset QA rows = 11873` 都是**错的**（实际评了 5,928 行）。

**修的时候有个坑**：`QARecord(question, answers[0], ...)` 对空答案会 `IndexError`，所以不是
把 127 行改成 `if filter_no_qa and (not question or not answers): continue` 就完事 —— 还要让
`QARecord.answer` 可空（评测打分读的是 `record.references` 那个 tuple，但构造和别的调用点会碰
`.answer`）。评测侧的"按 miss 计"已经天然支持：`evaluator.py:216-218` 对没有 active label 的行
`samples += 1` 但不加分；`metrics.py:125` 的 `references or ("",)` 还顺带实现了
**空输出（弃答）记 EM/F1=1** 的 SQuAD v2 口径。

**这一列现在能不能报？** 模型是在 `filter_no_qa=True` 上训的（从没学过弃答），所以即使把数据修好，
v2all 也只会退化成"5,945 行全 0"：预计 ON AR em **0.2530** / f1 **0.3200**、OFF **0.2022 / 0.2688**。
这是一个定义清楚的**下界**，可以报，但必须注明"模型不允许弃答"。想要真的 SQuAD v2 数字，
得先有弃答能力（对应还没做的 H5：无答案原文训练）。

**建议**：在修好之前，v2all 列改名为 `v2.0 (answerable only)` 或直接从脚本里去掉；对外只报
v1 + v2(answerable) 这两列（它们现在是干净的）。

## 2. 【意外但不影响数字】实际跑的是 8 卡，不是 4 卡

* 六份日志都是 `ranks=8`；v1 是 9 个 batch / 8 rank（rank0 两批），v2 是 5 个 batch / 8 rank。
* `accelerate` 只会打印 `More than one GPU was found ... pass --num_processes=1` 这种警告
  （条件是"没有显式传 `--multi_gpu`"），说明 `--num_processes` 是传了的；而
  `scripts/env.local.sh` 里 `export NUM_PROCESSES=` 两行都是注释 → **是 shell 里继承的
  `NUM_PROCESSES=8`**（训练那轮的 `export` 还在同一个 tmux 里），盖过了 `eval_squad_v1v2.sh` 的默认值 4。
* 为什么无害：这次 `max_qa=None`（全量评，没有 AR 行预算），解码是贪心（确定性），指标是逐行均值 ——
  `accelerate` 为了对齐 batch 数而复制的那些 batch 是**同样的行**，重复元素的均值不变；TF 的
  token/样本加权同样等比例。
* 什么时候会变成问题：一旦给评测加回 AR 上限（`autoregressive_max_qa` 是**全局行**语义），rank 数
  就会改变被评测的子集 → 那时必须显式写死 `NUM_PROCESSES`。另外 v2 只有 5 个 batch，8 卡里有 3 张
  在跑重复 batch：更快，但浪费。

## 3. 【口径，不是 bug】训练期数字与全集数字系统性差 ~0.03

| | 训练期 val（AR 曲线终点） | 全集 v1.1 | Δ |
| --- | --- | --- | --- |
| ON AR f1 / em / first_token_em | 0.6737 / 0.5342 / 0.6953 | 0.6445 / 0.5150 / 0.6615 | −0.029 / −0.019 / −0.034 |
| OFF AR f1 / em / first_token_em | 0.5716 / 0.4297 / 0.6250（step 3600） | 0.5448 / 0.4108 / 0.5995 | −0.027 / −0.019 / −0.026 |

原因：训练期 val 是 **v1+v2 混合的 16,498 行**、先取前 2000 个 context、AR 只评前 1024 行；
全集评测是 v1/v2 **分开**、`validation_max_samples=null`、AR 评**全部行**。两臂同向同幅，
所以 A/B 结论不变，但报告里不能把 0.6737 当成"全集 SQuAD F1"，它是"混合版本、前 1024 行"的数。

## 4. 【口径，建议写进文档】TF EM < AR EM、TF F1 > AR F1

四个子集里都是同一个形态（ON v1：TF em 0.4127 < AR em 0.5150，但 TF f1 0.6842 > AR f1 0.6445）。

原因在 `evaluator.py::teacher_forced`：TF 的预测文本是"**gold 答案每个位置的 argmax**"拼起来的
（`predictions[i][active]`，`active` 就是 gold 答案的 label 位置），长度被钉死在 gold 答案长度上、
**不可能提前停**；而 AR 是自己生成、`answer_line` 取第一个非空行，模型学会停就停。于是 TF 的 EM 被
系统性惩罚（多预测一个字就 0），F1 反而因为"部分命中"偏高。结论：**TF 的 em/f1 不能和 AR 的比大小，
TF em 也不是上界**；TF 的可用信号是 `qa_loss` / `ppl` / `reconstruction_loss` / `first_token_em`。

好消息（这条反而是校验通过）：同一子集里 TF 与 AR 的 `first_token_em` 差 ≤ 0.0004
（ON v1: 0.661892 vs 0.661511；OFF v1: 0.599971 vs 0.599542）——两个 pass 的答案起点编号与检索
判定完全一致，`9110877`（train == generate 位置编号）是对的。

## 5. 【卫生】数据集缓存每跑一次就作废，目录在堆积

同一个 v1 split（path 和 size 完全相同，6422978 B）在 `eval_on` 和 `eval_off` 下各生成了 **4 个**
缓存目录，metadata 里只有 `mtime_ns` 不同：

```
validation-8bf6886c...  eval_on/v1/squad/validation.jsonl  size=6422978  mtime_ns=1789662259387868149
validation-b5b6bf11...  eval_on/v1/squad/validation.jsonl  size=6422978  mtime_ns=1789662164225858134
validation-d254ef0d...  eval_on/v1/squad/validation.jsonl  size=6422978  mtime_ns=1789662054408566043
validation-e9a1ba56...  eval_on/v1/squad/validation.jsonl  size=6422978  mtime_ns=1789662108395358466
```

`source_files_metadata()`（`src/dataset_cache.py:14`）把 `mtime_ns` 算进指纹，而 `eval_squad_v1v2.sh`
每次都重写 split 文件 → 指纹每次都变 → 重建缓存（日志里的 `Reusing` 只在同一次运行的多个 rank
之间成立，不是跨 run）。目前已经累积 14 个 `validation-*` 目录。可选修法：指纹用内容 hash，或者
写 split 文件前先比较内容、内容相同就不动文件。

## 6. 建议的下一步

1. **先改表**：把 v2all 列改名/摘掉，只报 v1 + v2(answerable)；这两列可以直接用。
2. **要真 v2 数字**：修 `src/data.py`（让无答案行按配置进库，`QARecord` 支持空答案），并决定是否做
   弃答能力（H5）。在此之前只能报第 1 节算出的那个**下界**。
3. **和 ICL 基线比时**：把基线指到评测脚本已经写好的同一份 split 文件上，例如
   `--squad-validation-file outputs/eval_on/v1/squad/validation.jsonl --max-context-tokens 1024`
   —— 直接读 `aggregated/squad/validation.jsonl` 会把 v1 和 v2 的行混在一起（22,443 行），
   与 memory 侧任何一列都不可比。
4. **固定评测口径**：AR 为头条、全集（全部行、`validation_max_samples=null`）、显式
   `NUM_PROCESSES=4`，并把 rank 数记进 `results.md`（现在只有日志里有）。
