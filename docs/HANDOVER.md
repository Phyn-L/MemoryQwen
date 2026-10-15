# 交接文档：本轮 MemoryQwen 改造（数据链、配置分层、评估与训练启动）

> 面向接手本仓库的 Agent。**工作区尚未提交**：全部改动都在 `/data/lz/MemoryQwen` 的工作树里
> （`git status`：87 修改 / 22 删除 / 18 新增，`git diff --stat` 约 +3.6k/−83k 行）。
> 远端主机 `lz@10.24.116.90:1172`，conda 环境 `shine`（`source /home/lz/miniconda3/etc/profile.d/conda.sh && conda activate shine`）。
> 本文档写作时的验证状态：`python -m pytest tests` → **251 passed**；`python -m ruff check .` → **0 error**。

## 0. 30 秒速览（先读这段）

- **数据分两层**：`contexts/1qa/meta/` = 原始未过滤（`build_1qa.py`，不依赖配置）；
  `contexts/1qa/cache/{train,validation}.jsonl` = 按训练 YAML 过滤后的产物（`build_cache.py`，训练启动会自动补建）。
- **loader 不再重复过滤**：只要 cache 指纹（源 meta 文件 size/mtime + tokenizer + 过滤规则）匹配，
  日志出现 `[data] train: … (loader-side re-filter skipped)` 后直接读文件。
- **sortish 长度不再单独扫一遍**：过滤时顺手记进 `cache/<split>.jsonl.lengths.json`，`SortishSampler` 直接复用。
- **评估使用固定种子的分层抽样、只贪心**：没有 reservoir 抽样，也没有 temperature/top-p/beam；QA 按 `training.seed` 对每个数据集独立抽取至多 N 条，无合计条数上限，Recon 复用其涉及的去重 context。
- **配置分三层**：`harness.yaml`（共享）+ `models/qwen3-*.yaml`（模型）+ 实验 YAML（只写差异，用 `_base_` 组合）。
- **本轮遗留可做项**（第 8 节）：懒加载偏移索引（省训练启动 20 s / 5.2 GB）、mypy+coverage+GPU CI、把改动分批提交。

---

## 1. 仓库与运行环境

| 项 | 值 |
|---|---|
| 仓库根 | `/data/lz/MemoryQwen`（远端 `lz@10.24.116.90:1172`） |
| Python | conda `shine`，torch 2.13.0+cu132，transformers 5.16.1（`requirements.lock` 已提交） |
| 数据根 | `DATA_ROOT=/data/lz/MemoryQwen/contexts`（`utils/machines.py` 的 `machine: 4090`；h200/a800 各有一条） |
| HF cache | `/data/lz/hf_cache`（新式 `hub/datasets--*/snapshots/.../parquet` + SQuAD 的旧式 `datasets/squad/plain_text/0.0.0/*.arrow`） |
| 测试 | `python -m pytest tests`（CPU，约 13 s，251 passed） |
| Lint | `python -m ruff check .`（line-length 120；`docs` 与 `.tmp_analysis` 排除） |
| CI | `.github/workflows/ci.yml`：CPU torch + `ruff check .` + `pytest -q`（`PYTHONHASHSEED=0`）。mypy 有配置但**未进 CI**。 |

当前真实数据（已经用新链路重建过）：

```
contexts/1qa/
  meta/meta_train.json            898,260 contexts / 1,154,712 QA pairs
  meta/meta_validation.json        93,026 / 129,085
  meta/meta_test.json              92,787 / 128,676
  meta/{train,validation,test}/<slug>.json     9 个数据集各自一份
  meta/build.manifest.json        requested_split / actual_source_split / 计数
  cache/train.jsonl               845,151 contexts（1.61 GB）+ train.jsonl.lengths.json（3.9 MB）
  cache/validation.jsonl           74,001 contexts（95.8 MB）+ validation.jsonl.lengths.json
  cache/build.json                过滤指纹（判断是否需要重建）
  cache/loader/                   loader 派生缓存（排序长度缓存；过滤结果缓存）
  code/                           旧构建脚本备份（仅参考，已不在 scripts/ 里）
```

旧的 `1qa-train.jsonl` / `1qa-validation.jsonl` / `1qa-test/` / `1qa-val/` 已删除；`src/eval_data.py`
为此保留了回退路径（见 4.3），所以旧数据树也不会突然报错。

---

## 2. 本轮改动清单（按主题，含动机/文件/验证）

### 2.1 Recon 任务的 task embedding 初始化（本轮最早的一项）

- **动机**：原来 `recon_task_embedding` 是随机初始化，浪费了一个已知可用的先验。
- **做法**：用 tokenizer 对 `<|im_start|>assistant\n` 编码后的若干 token 的**输入 embedding 均值**初始化。
  - `src/model.py`：新增 `assistant_header_token_ids(tokenizer)`（`tokenizer.encode(..., add_special_tokens=False)`），
    模型构造函数接收 `task_init_token_ids`；找不到该 header 或 id 越界时回退到 Qwen3 默认 `(151644, 77091, 198)`；
    合成小词表仍走随机初始化。
  - `load_model` 负责把 `assistant_header_token_ids(tok)` 传进去。
- **验证**：`tests/test_context_writer.py` 三条路径（自定义 id、越界回退、helper 行为）。

### 2.2 配置分层（`harness.yaml` + `models/*.yaml` + 薄实验文件）

- **动机**：原来每个实验 YAML 复制了全套模型/机器参数，换模型或改机器要动十几份文件。
- **现状**：

  | 文件 | 内容 |
  |---|---|
  | `configs/harness.yaml` | 数据过滤/DataLoader、`loss`、`optimizer`、`scheduler`、`training`、`evaluation`、`logging`、`checkpoint`；**没有** `model`/`memory`，不能单独运行 |
  | `configs/models/qwen3-{1.7b,4b,8b}.yaml` | 只有 `model:`（权重路径、dtype、LoRA、`attn_implementation`） |
  | `configs/train_baseline.yaml` | `_base_: [harness.yaml, models/qwen3-1.7b.yaml]` + `memory:` 架构 |
  | `configs/h200_train_baseline.yaml` | `_base_: [train_baseline.yaml]` + h200 尺寸差异 |
  | `configs/train_baseline_qwen3_{4b,8b}.yaml` | `_base_: [train_baseline.yaml, models/qwen3-Nb.yaml]` + 实验差异 |
  | `configs/ablations/_ablation_common.yaml` | 消融共享层 |
- **验证**：`configs/train_baseline.yaml` 解析结果与改造前**逐字段一致**（当时比过：diff 为空）；
  新增测试断言分层不变量（harness 无 `model`/`memory`、models 只有 `model`、每个可运行预设都同时具备架构与模型）。

### 2.3 梯度累积

- **动机**：显存不够时不应缩小有效 batch。
- **做法**：`training.gradient_accumulation_steps`；`scripts/train.py` 用 `Accelerator(gradient_accumulation_steps=…)`
  + `accelerator.accumulate(model)`，无 accelerate 时手动累积并 `(total / accumulation).backward()`（因为
  `Accelerator.backward` 内部已经除过，只有 DeepSpeed 例外）。纯函数 `optimizer_steps()` / `micro_batch_skip()`
  负责"优化器步 ↔ micro-batch"换算；调度、评估、checkpoint 全部按**优化器步**计；日志记该组 micro-batch 的**均值**。
- **验证**：`tests/test_gradient_accumulation.py`（累积 == 合并 batch、短组仍平均、步数/跳过量、配置校验）；
  实机 smoke：4 个 micro-batch + accum=2 → `steps=2`、`accum=2 effective_batch=2`，loss 正好是两组均值。

### 2.4 生成路径重写（贪心、批量、显存友好）

- **动机**：原实现每步 `torch.cat` attention mask、按行 Python 记账，长前缀时又慢又吃显存。
- **做法**（`src/model.py::_generate_answers_with_prefix`）：
  - `step_mask` 预分配（`[count, prefill + max_new_tokens]`），前 `prefill` 列写死 `cache_prefix_mask`，每步只切片；
  - 无 padding 时直接 `attention_mask=None`；
  - token 记账向量化（`generated[count, max_new_tokens]` + `lengths`），结束时 `result[:, :width_out]`；
  - **只实现贪心**（`argmax` + EOS 停），文档化；`_select_cache` 对 `DynamicCache(…)` 构造失败给出带版本信息的报错。
- **验证**：`tests/test_generation.py`：批量 == 逐行 == 每批 1 行（含左 padding 行）；EOS 记账；
  `torch.multinomial` 被打成"调用即失败"后仍能生成（证明无采样）；源码守卫禁止 `do_sample=True`/`temperature=`/`top_p=`/`num_beams=`。

### 2.5 `_transformer_body`（无 LM head 的解码器主体）

- **动机**：`encode_context_prefix` / 蒸馏 student / `loss_only=True` 只需 `hidden_states`+KV；调 `self.qwen`
  会把每个位置投影到 151936 词表（白算 + 一行 ~0.6 GiB logits）。
- **做法**：`_resolve_transformer_body()` 解析一次并**用 `object.__setattr__` 缓存**（不能当普通属性，否则会注册成
  第二个 submodule、`state_dict()` 里整套骨干权重翻倍）；解析失败抛 `RuntimeError`（含类名与 transformers 版本），
  **不再静默回退**到完整模型。
- **验证**：`tests/test_generation.py::test_the_decoder_body_has_no_lm_head_and_is_not_a_second_submodule` 等。

### 2.6 评估窗口化（不再"整split遍历 + 跳过"）

- **动机**：旧字段 `evaluation.evaluation_max_samples` 是全局预算；现在 `evaluation.max_eval_sample_per_dataset` 为每个数据集独立限额，再按 rank 切分。旧实现让每个 rank 迭代**整个**
  split 再跳过窗口外的 batch —— 而 collate 在 loader 里，跳过前就已经读过并 tokenize 了，白白几分钟 IO。
- **做法**：
  - `src/pipeline.py`：按 dataset 累计 QA turn 配额、仅保留命中的上下文/turn，再将入选序列切给各 rank；
    Recon 仍独立按 context 上限窗口化。两个 loader 都**不交给 `accelerate.prepare`**。
  - `src/evaluator.py`：`autoregressive(..., row_base, pre_windowed)`、`recon_teacher_forced(..., context_base, pre_windowed)`、
    `compare_reader_tasks(..., row_base, recon_row_loader, recon_base, pre_windowed)`。
  - `scripts/train.py` / `scripts/test.py`：为两个预算（QA 行 / recon context）各建一个预窗口 loader。
- **验证**：`tests/test_eval_windows.py`（行索引、窗口→context 映射、loader 只取窗口、非法 unit）；AB 套件实测输出与改造前一致（`recon_ce=11.921361`）。

### 2.7 数据构建链重写成两步

- **删除**：`scripts/data/build_1qa_from_hf_cache.py`、`build_clean_1qa_splits.py`、`build_1qa_test.py`、
  `split_1qa_eval_sets.py`（四个互相 import 的脚本）。
- **新增** `scripts/data/build_1qa.py`（原始层）：
  - 三种来源：新式 hub parquet、SQuAD 的旧式 arrow IPC 流、MuSiQue 的 jsonl；
  - 约定：**一行一个 context + 它全部 `qa_pairs`**；不做 context 长度上限、不做每数据集条数截断；只丢空 context /
    空 question / 没有任何非空答案的 QA；**MS MARCO 只用 `is_selected=1` 的 passage**；context 只来自本行文本；
    划分缺失或没有可用行时按 `test→validation→train` 回退并记录 `actual_source_split`；
  - 产物：`meta/meta_<split>.json`（JSONL 内容、`.json` 扩展名）+ `meta/<split>/<slug>.json` + `meta/build.manifest.json`；
  - 多进程：任务粒度 = (数据集, 划分)，父进程按数据集顺序写盘 → 串行/并行**逐字节一致**（有测试）。
- **新增** `scripts/data/build_cache.py` + `src/cache_build.py`（过滤层）：
  - 复用**加载器同一个过滤函数** `src/data.py::filter_rows_batch`（单一规则来源）；
  - 产物：`cache/train.jsonl`、`cache/validation.jsonl`、`cache/build.json`（指纹）、`cache/<split>.jsonl.lengths.json`；
  - 指纹 = {version, split, 源 meta 的 size/mtime, tokenizer 指纹, 数据集选择, 过滤选项}；
  - 多进程：分块（行数 512 + 字节 8 MiB 双上限）发给 worker，**按序回收** → 与串行逐字节一致；
  - 不在构建时按条数截断（`train_max_samples`/`validation_max_samples` 是 loader 侧的确定性上限）。
- **验证**：`tests/test_build_1qa.py`、`tests/test_build_cache.py`；真实数据 dry-run 与历史语料比对：
  validation **129,085 QA 完全一致**；train QA 1,154,712 完全一致、context 少 15,946 且**全部集中在 msmarco**
  （正是"只用 selected passage"造成的），其余 7 个数据集逐项相同。

### 2.8 固定种子评估抽样 + 只贪心（删掉 reservoir 采样）

- **删除**：`src/data_sampling.py` + `tests/test_data_sampling.py`；配置字段 `validation_max_samples_per_dataset`、
  `validation_sampling_workers`、`validation_source_file`、`validation_output_dir`。
- **做法**：验证集就是 `cache/validation.jsonl`；QA 使用 `training.seed` 按数据集独立随机抽取至多 N 条（`max_eval_sample_per_dataset`），无 QA 总数上限；Recon 复用这些 QA 涉及的去重 context。
  多卡切分与卡数无关，入选 QA 和指标结果可复现。旧 `1qa-val/` 仅作 `eval_data` 的回退。
- **配置字段迁移报错**：`utils/config.py::REMOVED_DATA_FIELDS` 对这四个字段给出**具名**错误与替代方案；
  已保存的旧 checkpoint 里若带这些字段，`scripts/test.py` 用 `tolerate_removed_data_fields=True` 打印 warning 后丢弃
  （否则老 checkpoint 会变得无法评测）。
- **解码**：`src/model.py` 贪心；ICL baseline `generate(..., do_sample=False)`；两个 tripwire 测试见 2.4。

### 2.9 多进程（构建侧）与实测数据

- 之前：只有 loader 侧有 `ProcessPoolExecutor`；两个构建脚本**完全单进程**。
- 现在：两侧共用**一个驱动** `src/data.py::filter_row_chunks(chunks, tokenizer, options, num_workers, with_lengths)`。
- **实测（本机 112 核，真实数据）**：

  | 场景 | 串行 | 2 | 8 | 16 |
  |---|---|---|---|---|
  | filter（20k 行真实 meta） | 581 row/s（898k 行≈25.8 min） | 1315（11.4 min） | 2608（5.7 min） | 2833（5.3 min） |
  | sortish 长度（20k 行） | **14,224 ctx/s（845k 行≈1.0 min）** | 4,433 | 9,985 | — |

  结论：**过滤吃进程池**（8~16 后饱和）；**sortish 长度故意保持单进程**（HF tokenizer 内部已多线程，
  fork+pickle 整段 context 更贵）——这一条在 `scripts/train.py` 与 `configs/README.md` 有注释记录。
- **注意**：池用默认 `fork`，父进程可能已持有 tokenizer/torch 线程 → Python 3.12 的
  `DeprecationWarning: … multi-threaded, use of fork() …`。已写进 `pyproject.toml` 的 `filterwarnings` 并注明
  缓解办法是 `multiprocessing.set_start_method("forkserver")`。

### 2.10 进度条（三处）

1. `build_cache`：`Filtering meta_train.json` 按**字节**推进，postfix `rows/contexts/qa_pairs`；`--no-progress`/`--quiet` 关闭，非主进程不画。
2. `build_1qa`：没有条，改为每个 (数据集, 划分) 打印一行 `[read]/[skip]`（27 个任务时比进度条信息量大）。
3. loader：`ConversationDataset.READ_BAR_MIN_BYTES = 64 MiB` —— `pre_filtered` 读大文件时显示
   `Reading train.jsonl`（字节总量 + `rows=` postfix）；带 `--max-samples` 的按需读不画也不整读。
4. 训练 tqdm postfix：**一次** `set_postfix(step_postfix(group_terms, group_micro_batches))`，
   字段 `loss / qa / recon / ppl`（组均值；`ppl = exp(qa_loss)` clamp exp(20)）。
   位置参数传有序 dict，因为 tqdm 会把 `**kwargs` 按字母序排。

### 2.11 上下文长度 sidecar（"长度从过滤里免费带出"）

- `filter_rows_batch(rows, tokenizer, options, lengths_out=None)`：判 `max_context_tokens` 时已经算出的 token 数，
  对通过的行追加（`-1` = 没量，例如 `filter_long_context: false`）。**保留行的真实长度 == 当时算出的截断长度**
  （超限行已被丢弃）。
- `filter_row_chunks(..., with_lengths=True)`：worker 返回 `(rows, lengths)`；顺序保持。
- `build_cache` 写 `cache/<split>.jsonl.lengths.json` = `{version, size, mtime_ns, tokenizer, max_context_tokens, lengths}`；
  `--no-lengths` 关闭；`--lengths-only` 只为**已存在**的 cache 补写（不重过滤）。
- `src/data.py::read_context_lengths(dataset, tokenizer)`：校验 size/mtime、tokenizer、上下文上限、version、
  长度数 ≥ `len(dataset)`，命中则返回前缀。`SortishSampler._load_or_build_lengths` 优先用它，打印
  `[data] reusing N context lengths recorded next to … (no sortish pass)`；否则照旧测量（有条 + 缓存）。
- `ensure_cache` 会为"新鲜但缺 sidecar"的 cache 自动补写一次（**绝不重过滤 GB 数据**）。
- **实测（真实数据）**：补写 train 845,151 条 = 1m13s、validation 74,001 条 = 4s（一次性）；
  之后 sampler 长度 **0.88 s**（之前约 1.5 min），且 sidecar 与重新测量**逐项相同**（含前 5000 条独立重算）。
  长度统计 min/median/max = 4 / 109 / 1536。以后重建 cache 时长度是**免费**的（有 `--force` 重建验证）。

### 2.12 loader 派生缓存统一到数据树

- 之前 `pipeline.data_cache_dir()` 硬编码 `Path("outputs")/"conversation_cache"`。
- 现在 `data_cache_dir(cfg)` 默认 `<data.root>/1qa/cache/loader`，可用 `data.preprocess_cache_dir` 覆盖
  （数据根是慢网盘时指到本地快盘）。`outputs/conversation_cache` 已不再产生。
- 里面对两类文件：`conversation_filtered-<指纹>.jsonl`（只有"该文件不是新鲜 cache"时才写，例如 `--data-file` 原始评测文件、
  改过过滤参数、test split）与 `sortish_lengths-<hash>.json`（分桶长度）。

### 2.13 清理与一致性

- **删除**：`docs/experiments/*`（7 md + 3 JSON）、`docs/history/*`、`docs/REPOSITORY_LAYOUT_PLAN.md`、
  `utils/config_paths.py`、`scripts/data/download_hotpotqa.py`、空配置树 `configs/4090|h200/`，
  以及上文的数据脚本；相应引用（`docs/README.md`、`README.md`、`docs/guides/EVALUATION.md`、`scripts/README.md`）已修。
- **实验 runner（15 个 `scripts/experiments/*.sh`）迁移**：基座改为 `TrainConfig.from_file(base).to_dict()`
  （原来 `yaml.safe_load` 拿不到 `_base_`，且往 `memory:` 塞目标权重会被新 schema 拒绝）→ 目标权重进 `loss:`，
  架构项留在 `memory:`；混合的 per-run 覆盖在循环里按键名分流。另外删掉 4 处残留的宿主路径调用
  （`exec` 之后的 `python /home/sean/burn.py`，实际不可达）、把 v2 writer 脚本里早已改名的
  `evaluation.autoregressive_max_qa` 修成当时的 `evaluation_max_samples`（现为 `max_eval_sample_per_dataset`）、把 phase3 的 `RUNS` 数组裁到它真正生成的 10–14。
  **验证**：逐个 `--dry-run`（13 个 OK；`run_cumulative_memory_ablation` 仍失败，原因见第 9 节）。
- **工程配置**：`pyproject.toml`（pytest/ruff/mypy 段 + fork 警告过滤）、`requirements.lock`、`.github/workflows/ci.yml`。

### 2.14 新增测试（18 个文件中的 8 个是新加的）

`tests/test_build_1qa.py`、`test_build_cache.py`、`test_eval_data.py`、`test_eval_windows.py`、`test_generation.py`、
`test_gradient_accumulation.py`、`test_loss_reduction.py`、`test_recon_metric.py`、`test_turn_sampling.py`
（另有 `test_config_env.py`、`test_all_suite_orchestration.py`、`test_conversation_dataset_cache.py`、`test_machines.py`、
`test_resume.py`、`test_train_schedule.py` 等被更新）。

---

## 3. 关键设计决策（改动背后的"为什么"，避免被后人误改回去）

1. **一个过滤实现**：loader 与 `build_cache` 都走 `filter_rows_batch`；两个 driver 用同一个 `filter_row_chunks`。
   历史上"两处各写一份规则"正是规则漂移的根源。
2. **cache 位置与指纹**：`cache/build.json` 的指纹决定 `uses_filtered_cache()`；命中就 `pre_filtered=True` 直接读，
   不命中才回退过滤。**不要**为了"省事"绕过指纹检查。
3. **长度 sidecar 的合法性依赖 `filter_long_context`**：只有在该开关打开时，"截断计数 == 真实长度"成立；
   关掉时 filter 记 `-1`，`build_cache` 不写 sidecar，sampler 自己测量。
4. **sortish 长度保持串行**：实测并行更慢（见 2.9），不要"顺手"加 `num_workers`。
5. **过滤池用 fork 且刻意如此**：输出已被测试钉成与串行逐字节一致；若遇到挂住，改 `forkserver` 而不是删池。
6. **评估固定种子抽样、只贪心**：抽样使用 `training.seed`，相同配置得到相同 QA 与 Recon contexts；生成使用 `argmax`。
7. **训练 tqdm 一次 set_postfix**：tqdm 的 `set_postfix` 是替换而非合并；两次调用会互相抹掉（这就是"ppl 只剩 loss"的根因）。
8. **数据字段迁移要报名字**：`REMOVED_DATA_FIELDS` + checkpoint 宽容加载，避免老 run 直接崩。
9. **不要给 `_transformer_body` 用普通赋值**：会注册重复 submodule。

---

## 4. 数据流水线（现状）

### 4.1 全链路

```
HF cache ──build_1qa.py──► contexts/1qa/meta/{meta_<split>.json, <split>/<slug>.json, build.manifest.json}
                                        │
                    build_cache.py（=训练配置的过滤规则，多进程，带进度条）
                                        ▼
              contexts/1qa/cache/{train.jsonl, validation.jsonl, build.json, *.lengths.json}
                    │                                  │
        训练（train.sh / train.py）                 评测套件（meta/<split>/<slug>.json）
                    │
        loader 派生：cache/loader/{conversation_filtered-*.jsonl（仅在需要时）, sortish_lengths-*.json}
```

### 4.2 谁读什么

| 消费方 | 文件 | 说明 |
|---|---|---|
| `train.py` | `cache/train.jsonl`、`cache/validation.jsonl` | `data.split_file("train"\|"validation")`；`--data-root` 可覆盖 |
| `scripts/test.py --split test` | `meta/meta_test.json` | 没有"过滤后的 test"产物；`--data-file` 可覆盖 |
| AB 套件 `test_all_suite.py` | `meta/{test,validation}/<slug>.json` | 每数据集一份；`src/eval_data.resolve()` 是唯一决策点 |
| ICL 套件 | 同上 + `cache/train.jsonl`（few-shot 拆样本） | `src/icl_baseline.iter_examples` 支持 `conversations` 与 `qa_pairs` 两种方言 |
| `SortishSampler` | `cache/<split>.jsonl.lengths.json` | 命中则不测量 |

### 4.3 兼容/回退

- `src/eval_data.py::SPLIT_DIRS` 指向 `meta/<split>/`；`FALLBACK_SPLIT_DIRS` 保留
  `1qa-test/*.jsonl`、`1qa-validation/*.jsonl`、`1qa-val/*.jsonl`，所以旧数据树仍能评测。
- `data.files.<split>_conversation_file` 显式覆盖永远优先；此时 `uses_filtered_cache()` 为 False → 走逐行过滤 + `cache/loader` 缓存。
- `1qa-val/`（旧的抽样小子集）已无生产者；只在评测解析时兜底。

---

## 5. 配置系统要点

- 三层组合：`_base_: [harness.yaml, models/qwen3-1.7b.yaml]`，深度合并；`${VAR:-default}` 环境展开；
  `machine:` 从 `utils/machines.py` 表里选 `DATA_ROOT`/`MODEL_ROOT`（一条命令里 `--machine 4090`）。
- `data` 分组（`files`/`limits`/`loader`/`task`）会被 `_flatten_data_groups` 展开成扁平字段；展开时
  **字段感知**：`DataConfig` 里存在的键即使值是 mapping 也保留；未知键报错；`REMOVED_DATA_FIELDS` 里的键报具名错误。
- 目标函数字段在 `loss:`（`qa_weight`/`recon_weight`/`distill_*`/`max_logits_rows`），**不再**在 `memory:`；
  老配置里塞在 `memory:` 会被明确拒绝并提示"moved to the loss section"。
- `data.limits` 只有一处：`max_context_tokens`（过滤上限，真实数据用 1536）、`max_input_tokens`（模型窗口）、
  `max_question_tokens`/`max_answer_tokens`/`max_conversation_tokens`。
- `training.gradient_accumulation_steps`；`evaluation.max_eval_sample_per_dataset`（每数据集 QA 抽样上限，无合计上限；Recon 复用抽样 context）。
- `data.preprocess_cache_dir`（loader 派生缓存位置）、`data.cache_filtered_dataset`、`data.cache_sortish_lengths`。

---

## 6. 常用操作手册（runbook）

```bash
# 环境
source /home/lz/miniconda3/etc/profile.d/conda.sh && conda activate shine
cd /data/lz/MemoryQwen

# 测试与 lint（接手第一件事）
python -m pytest tests          # 期望 251 passed
python -m ruff check .          # 期望 0 error

# 1) 原始数据（不依赖配置；全量约几分钟）
python scripts/data/build_1qa.py                       # 全部数据集 × train/validation/test → meta/
python scripts/data/build_1qa.py --datasets squad race --splits validation --dry-run

# 2) 过滤后的训练/验证集（规则来自训练 YAML；全量约 5 min with --num-workers 16）
python scripts/data/build_cache.py --config configs/train_baseline.yaml
python scripts/data/build_cache.py --config configs/train_baseline.yaml --splits validation --dry-run
python scripts/data/build_cache.py --config configs/train_baseline.yaml --lengths-only   # 只补长度 sidecar
python scripts/data/build_cache.py --config configs/train_baseline.yaml --force          # 强制重建
# 说明：train.sh 启动时会自动补建缺失/过期的 cache 与缺失的 lengths

# 3) 训练
bash scripts/train.sh --machine 4090 --config configs/train_baseline.yaml
# 启动日志顺序（正常）：
#   [cache] …（仅在需要重建时，带 Filtering 进度条）
#   [data] train: train.jsonl is this config's filtered cache (loader-side re-filter skipped)
#   Reading train.jsonl: 100%|…| 1.61G/1.61G [00:20, … rows=845151]
#   [data] reusing 845151 context lengths recorded next to … (no sortish pass)
#   [data] validation: … skipped / Reading validation.jsonl …
#   schedule: machine=4090 steps=… accum=… effective_batch=… output_dir=…
#   Training: … loss=…, qa=…, recon=…, ppl=…      ← postfix 四字段，不再被覆盖
# 当前这段启动约 23 s（之前约 2 min）

# 4) 评测
bash scripts/evaluation/test_all_suite.sh …            # 或 python scripts/evaluation/test_all_suite.py
python scripts/test.py --ckpt <ckpt> --split validation --data-file contexts/1qa/meta/validation/squad.json
python scripts/evaluation/test_icl_suite.py …          # ICL baseline（do_sample=False）
```

常见现象对照：

| 现象 | 原因 | 处理 |
|---|---|---|
| `[data] … (loader-side re-filter skipped)` 之后静默十几秒 | 父进程在读+JSON 解析 cache 文件（1.61 GB ≈ 20 s，峰值 5.2 GB） | 正常；现在有 `Reading …` 进度条 |
| `Preparing sortish lengths` 出现 | 该 cache 没有 lengths sidecar（旧 cache / `--no-lengths` / 换配置） | 一次性；跑 `build_cache.py --lengths-only` 或让它自己缓存 |
| `Filtering train.jsonl … / Reading train.jsonl …` 又出现 | cache 指纹不匹配（改过滤参数、换 tokenizer、`--data-file` 指原始文件） | 正常回退；想避免就重建 cache |
| `data.validation_* was removed: …` | 老配置字段 | 按提示删字段（评估不再抽样） |
| `unknown data field(s): squad` | 老配置里残留 `validation_max_samples_per_dataset` 子表 | 同上（直接删该键） |
| `RuntimeError: cannot locate the transformer body …` | transformers 包装结构变了 | 按报错里提示扩展 `_resolve_transformer_body` |
| NCCL `unhandled cuda error` | 4 张卡被别的任务占满（本机常见） | 等空闲或换机器；与代码无关 |

---

## 7. 未提交内容的建议切分

工作树包含 109 个文件的改动，建议按下面顺序分批提交（每批都能独立通过 `pytest` + `ruff`）：

1. `chore(quality)`: `pyproject.toml`、`requirements.lock`、`.github/workflows/ci.yml`、删 `utils/config_paths.py`、删旧 docs。
2. `refactor(config)`: 配置分层 + `REMOVED_DATA_FIELDS`/宽容加载 + 相关测试。
3. `feat(data)`: `build_1qa.py`/`build_cache.py`/`cache_build.py`/`eval_data.py`/`data.py` 过滤与 sidecar + 测试。
4. `feat(loader)`: `pre_filtered`、`data_cache_dir(cfg)`、`Reading` 进度条、`filter_row_chunks` 共用 driver + 测试。
5. `feat(train)`: 梯度累积、tqdm postfix、预窗口评估 loader + 测试。
6. `feat(model)`: 生成路径重写、`_transformer_body`、task embedding 初始化 + 测试。
7. `chore(experiments)`: 15 个实验 runner 的迁移与清理。
8. 数据目录（`contexts/`）不在 git 里，无需提交。

---

## 8. 尚未完成 / 建议的下一步（按性价比）

1. **懒加载 + 字节偏移索引**（去掉训练启动最后那 20 s 与 5.2 GB 常驻内存）：扫换行 0.9 s 建偏移表，
   `__getitem__` seek+parse，解析摊到 DataLoader worker 并行。风险：`len()`、`SortishSampler`、evaluator 迭代都要适配，
   索引要按 size/mtime 失效。约 1–2 小时。
2. **质量门禁**：mypy 只有配置没进 CI；无覆盖率门槛；无 GPU 冒烟（`.github` 只有 CPU job）。
3. **吞吐**：允许 `model.attn_implementation=flash_attention_2`（现在只允许 `sdpa|eager`）；`torch.compile` 只包训练步
   （先 profile 判断是否 launch/带宽 bound）；sequence packing 在本项目收益有限（sortish 分桶已缓解 padding 浪费）。
   详见上一轮讨论的结论。
4. **`run_cumulative_memory_ablation.sh`** 需要 `readout_length > 0` 的 base config（它自己的前置条件，不是 bug）。
5. 评估侧 `Preparing sortish lengths` 之外的**首次**开销：`cache_filtered_dataset` 对原始评测文件仍需过滤一次
   （每 (文件, tokenizer, 规则) 一次，缓存在数据树下）。

---

## 9. 环境相关的已知问题（不是代码问题）

- **GPU 被占用**：本机 4 张卡常被别的任务占到 22.7–23.4 GB / 24.5 GB。表现为 AB 套件 `torch.distributed` 报
  `NCCL error: unhandled cuda error` / rank 被 SIGTERM；CPU 冒烟在 `torch.optim` 的 CUDA health check 处 OOM。
  处理：等空闲，或给 CPU smoke 加 `CUDA_VISIBLE_DEVICES=""`。
- **shell**：远端默认 zsh —— `set -- $x` 不做分词；glob 无匹配会报错（要么引号要么 `find`）。
- **长时间任务**：用 `nohup … &` + 日志文件轮询；`rw_exec` 的默认超时会掐断前台命令。

---

## 10. 变更文件速查

- 新增：`configs/harness.yaml`、`configs/models/qwen3-{1.7b,4b,8b}.yaml`、`configs/ablations/_ablation_common.yaml`、
  `src/eval_data.py`、`src/cache_build.py`、`scripts/data/build_1qa.py`、`scripts/data/build_cache.py`、
  `requirements.lock`、`.github/workflows/ci.yml`、9 个新测试文件。
- 删除：4 个旧数据脚本、`src/data_sampling.py`、`utils/config_paths.py`、`scripts/data/download_hotpotqa.py`、
  `docs/experiments/*`、`docs/history/*`、`docs/REPOSITORY_LAYOUT_PLAN.md`、`tests/test_data_sampling.py`。
- 主要修改：`src/data.py`、`src/pipeline.py`、`src/model.py`、`src/evaluator.py`、`src/losses.py`、`utils/config.py`、
  `utils/checkpoint.py`、`utils/machines.py`、`scripts/train.py`、`scripts/test.py`、`scripts/evaluation/*`、
  `scripts/experiments/*`（15 个）、`configs/*`、`docs/guides/*`、`src/PROJECT_STRUCTURE.md`、`contexts/README.md`。

> 维护提醒：任何涉及"数据从哪来、规则是什么"的改动，先看 `src/eval_data.py`（评测文件位置）、
> `src/cache_build.py`（过滤与指纹）、`src/pipeline.py::uses_filtered_cache`（快路径判定）这三处，
> 它们各自是唯一的决策点。
