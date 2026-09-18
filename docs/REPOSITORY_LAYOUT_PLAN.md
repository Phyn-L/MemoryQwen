# 仓库目录整理提案

日期：2026-09-18。状态：第一阶段已执行（2026-09-18）；第二阶段未执行。范围：降低查找和理解成本。

## 1. 决策与边界

第一阶段只整理文档、增加导航和补齐缓存忽略规则；现有 Python 模块、测试、训练评测入口、配置及实验产物路径保持原位。第二阶段才考虑移动专用脚本，独立验证。模型拆分、Python 包重命名、配置继承系统和实验产物搬迁均不纳入本轮。

以下映射覆盖检查时所有 Git 已跟踪文件及非忽略的普通未跟踪文件；`.hf_cache/` 和已忽略的运行产物按目录整体保留，规则适用于其每个内部文件。没有把未跟踪文件当作可删除文件。表中“当前文件”保留整理前快照；第一阶段文档目标路径现已创建。

## 2. 第一阶段目标目录

```text
MemoryQwen/
├── README.md                    # 项目入口：方法摘要、快速运行、导航
├── pyproject.toml
├── .gitignore
├── src/                         # 模型、数据、损失、评测；保持全部现有文件
├── utils/                       # 配置、机器、checkpoint、优化器等；保持原位
├── scripts/                     # 第一阶段保持所有脚本路径
├── configs/                     # 保持现有模型规模分组和全部 YAML
│   ├── README.md                # 新增：配置用途和实验对应关系
│   ├── qwen-1.7b/
│   ├── qwen-4b/
│   └── qwen-8b/
├── tests/                       # 保持现有平铺结构
├── docs/
│   ├── README.md                # 新增：按问题导航、当前说明与历史记录的边界
│   ├── REPOSITORY_LAYOUT_PLAN.md # 本提案
│   ├── guides/                  # 当前操作及方法说明
│   │   ├── RUNNING.md           # 从 README 提取环境、训练、恢复操作
│   │   ├── EVALUATION.md        # 从 README 提取当前评测协议与指标语义
│   │   ├── METHOD.md            # 从 README 提取经代码核对的当前实现
│   │   └── READER_OPTIONS.md
│   ├── experiments/             # 实验设计、观察、结果与异常的时间快照
│   │   ├── README.md            # 实验索引：问题→配置→入口→证据→状态
│   │   ├── AB_H200.md
│   │   ├── ROUND3_PLAN.md
│   │   └── EVAL_ANOMALIES.md
│   ├── history/                 # 修复过程、旧设计与实施记录
│   │   ├── ENGINEERING.md
│   │   ├── IMPROVEMENTS.md
│   │   └── PLAN_reader_upgrade.md
│   └── visuals/                 # HTML 展示与可编辑图源；标明适用版本
│       ├── method_overview.html
│       ├── qwenmemory_overview.html
│       └── qwenmemory_method.drawio
├── outputs/                     # 保留：checkpoint、缓存、评测结果
├── logs/                        # 保留：运行日志
├── wandb/                       # 保留：W&B 运行数据
├── .hf_cache/                   # 保留：补充 Git 忽略
└── .tmp_analysis/               # 保留：临时分析与原始证据
```

`package.json` 和 `package-lock.json` 暂时仍在根目录，待核实用途后单独决定是否删除；不为两个空清单新建工具目录。两个 HTML 页面也不按文件名推断谁是最新版：保留两份，核对内容与代码版本后在索引中指定推荐入口。

## 3. 逐文件去向

“原位”表示目标路径等于当前路径。第二阶段列是后续候选，本次未执行。所有配置保持完整 YAML，不引入继承或同时重命名实验。

| 当前文件 | 第一阶段目标 | 第二阶段 / 处理说明 |
|---|---|---|
| `.gitignore` | 原位 | 增加 `.hf_cache/`，保留现有规则 |
| `ENGINEERING.md` | `docs/history/ENGINEERING.md` | 迁移并修复链接；核对状态/版本标记 |
| `IMPROVEMENTS.md` | `docs/history/IMPROVEMENTS.md` | 迁移并修复链接；核对状态/版本标记 |
| `README.md` | 原位 | 精简为入口；内容拆分规则见第 4 节 |
| `configs/4090/qwen-1.7b/reader/train_reader-off_ctx1024_m64.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-1.7b/reader/train_reader-on_ctx1024_m64.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-1.7b/context_length/train_reader-on_ctx2048_m64.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m16.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m8.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-1.7b/baseline/train_baseline.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-4b/baseline/train_baseline.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `configs/4090/qwen-8b/baseline/train_baseline.yaml` | 原位 | 配置内容与路径不变；纳入配置索引 |
| `docs/AB_H200.md` | `docs/experiments/AB_H200.md` | 迁移并修复链接；核对状态/版本标记 |
| `docs/EVAL_ANOMALIES.md` | `docs/experiments/EVAL_ANOMALIES.md` | 迁移并修复链接；核对状态/版本标记 |
| `docs/PLAN_reader_upgrade.md` | `docs/history/PLAN_reader_upgrade.md` | 迁移并修复链接；核对状态/版本标记 |
| `docs/READER_OPTIONS.md` | `docs/guides/READER_OPTIONS.md` | 迁移并修复链接；核对状态/版本标记 |
| `docs/ROUND3_PLAN.md` | `docs/experiments/ROUND3_PLAN.md` | 迁移并修复链接；核对状态/版本标记 |
| `docs/method_overview.html` | `docs/visuals/method_overview.html` | 迁移并修复链接；核对状态/版本标记 |
| `docs/qwenmemory_method.drawio` | `docs/visuals/qwenmemory_method.drawio` | 迁移并修复链接；核对状态/版本标记 |
| `docs/qwenmemory_overview.html` | `docs/visuals/qwenmemory_overview.html` | 迁移并修复链接；核对状态/版本标记 |
| `package-lock.json` | 原位 | 暂留；核实工具用途后再决定删除 |
| `package.json` | 原位 | 暂留；核实工具用途后再决定删除 |
| `pyproject.toml` | 原位 | 保持内容与接口 |
| `scripts/download_hotpotqa.py` | 原位 | 候选迁移至 `scripts/data/download_hotpotqa.py`；检查路径推导及调用者 |
| `scripts/archive/eval_icl_squad.sh` | 原位 | 候选迁移至 `scripts/evaluation/eval_icl_squad.sh`；旧 shell 入口保留转发兼容 |
| `scripts/archive/eval_squad_v1v2.sh` | 原位 | 候选迁移至 `scripts/evaluation/eval_squad_v1v2.sh`；旧 shell 入口保留转发兼容 |
| `scripts/archive/run_ab.sh` | 原位 | 候选迁移至 `scripts/experiments/run_ab.sh`；旧 shell 入口保留转发兼容 |
| `scripts/archive/run_on_8b.sh` | 原位 | 候选迁移至 `scripts/experiments/run_on_8b.sh`；旧 shell 入口保留转发兼容 |
| `scripts/archive/run_queue.sh` | 原位 | 候选迁移至 `scripts/experiments/run_queue.sh`；旧 shell 入口保留转发兼容 |
| `scripts/test.py` | 原位 | 保持内容与接口 |
| `scripts/test.sh` | 原位 | 保持内容与接口 |
| `scripts/test_icl_baseline.py` | 原位 | 保持内容与接口 |
| `scripts/archive/test_icl_baseline.sh` | 原位 | 保持内容与接口 |
| `scripts/train.py` | 原位 | 保持内容与接口 |
| `scripts/train.sh` | 原位 | 保持内容与接口 |
| `src/MemoryDecoder.py` | 原位 | 本轮不改大小写或导入路径 |
| `src/__init__.py` | 原位 | 保持内容与接口 |
| `src/data.py` | 原位 | 保持内容与接口 |
| `src/dataset_cache.py` | 原位 | 保持内容与接口 |
| `src/dtypes.py` | 原位 | 保持内容与接口 |
| `src/evaluator.py` | 原位 | 保持内容与接口 |
| `src/icl_baseline.py` | 原位 | 保持内容与接口 |
| `src/losses.py` | 原位 | 保持内容与接口 |
| `src/metrics.py` | 原位 | 保持内容与接口 |
| `src/model.py` | 原位 | 本轮不拆分模型 |
| `src/pipeline.py` | 原位 | 保持内容与接口 |
| `src/resampler.py` | 原位 | 保持内容与接口 |
| `tests/test_ab_configs.py` | 原位 | 保持内容与接口 |
| `tests/test_ae_lm.py` | 原位 | 保持内容与接口 |
| `tests/test_config_env.py` | 原位 | 保持内容与接口 |
| `tests/test_distill.py` | 原位 | 保持内容与接口 |
| `tests/test_eval_logging.py` | 原位 | 保持内容与接口 |
| `tests/test_eval_protocol.py` | 原位 | 保持内容与接口 |
| `tests/test_icl_data.py` | 原位 | 保持内容与接口 |
| `tests/test_lora_dtype.py` | 原位 | 保持内容与接口 |
| `tests/test_machines.py` | 原位 | 保持内容与接口 |
| `tests/test_masks.py` | 原位 | 保持内容与接口 |
| `tests/test_memory_init.py` | 原位 | 保持内容与接口 |
| `tests/test_metrics.py` | 原位 | 保持内容与接口 |
| `tests/test_positions.py` | 原位 | 保持内容与接口 |
| `tests/test_readout.py` | 原位 | 保持内容与接口 |
| `tests/test_references.py` | 原位 | 保持内容与接口 |
| `tests/test_resume.py` | 原位 | 保持内容与接口 |
| `tests/test_tied_backbone.py` | 原位 | 保持内容与接口 |
| `tests/test_train_schedule.py` | 原位 | 保持内容与接口 |
| `tests/test_vocab_head.py` | 原位 | 保持内容与接口 |
| `utils/__init__.py` | 原位 | 保持内容与接口 |
| `utils/checkpoint.py` | 原位 | 保持内容与接口 |
| `utils/config.py` | 原位 | 保持内容与接口 |
| `utils/ddp.py` | 原位 | 保持内容与接口 |
| `utils/machines.py` | 原位 | 保持内容与接口 |
| `utils/optimizer.py` | 原位 | 保持内容与接口 |
| `utils/scheduler.py` | 原位 | 保持内容与接口 |

### 本地目录与元数据

| 当前路径 | 去向与规则 |
|---|---|
| `outputs/**` | 全部原位；不移动 checkpoint、数据缓存、评测结果，不更改恢复路径 |
| `logs/**`、`wandb/**` | 全部原位；继续忽略 |
| `.hf_cache/**` | 全部原位；只增加忽略规则 |
| `.tmp_analysis/**` | 全部原位；整理文档不能删除其引用的原始证据。需长期复用的分析脚本另行审阅后再提升为正式脚本 |
| `.pytest_cache/**`、各处 `__pycache__/**` | 保持忽略；本轮不清理 |
| `.vscode/settings.json` | 原位；保持本地编辑器配置 |
| `.git/**`、`.agents/**`、`.codex/**` | 原位；不参与整理 |
| `scripts/env.local.sh`（若本机存在） | 原位并保持忽略；全部启动脚本继续从此处读取 |
| `package.json`、`package-lock.json` | 当前分别为空对象、空 packages 锁文件；待核实用途，不自动删除 |

## 4. 文档内容如何拆分

| 来源 | 目标与处理 |
|---|---|
| README 项目摘要、最小安装训练命令 | 根 README 保留；补充通往评测、实验和方法的链接 |
| README “Running on another machine”、恢复路径及缓存说明 | `docs/guides/RUNNING.md`；保留优先级和机器配置规则 |
| README answer-target、metric semantics、SQuAD 数据口径 | `docs/guides/EVALUATION.md`；当前规则与历史问题区分，避免把旧指标当当前结论 |
| README dtype、LoRA、prefix sharing、memory objectives、数值不变量 | `docs/guides/METHOD.md`；先核对当前代码，不将旧说明直接标为最新 |
| README 已修复 bug 的过程与历史数值 | `docs/history/ENGINEERING.md` 的带来源补充节；重复内容合并，保留证据及适用 commit |
| README 未实现扩展想法 | `docs/history/PLAN_reader_upgrade.md` 的独立“未实施想法”节；不混入已实现状态 |
| `READER_OPTIONS.md` | 使用说明进入 guides；历史冒烟结果移入历史实施记录，现行开关按代码核对 |
| AB、第三轮、异常记录 | 保留实验事实与当时命令；增加日期、代码版本、协议版本和状态，不擅自把预测改成结果 |
| 两份 HTML 与 drawio | 保留内容，补来源与适用版本；HTML 相对链接需要随迁移检查 |

新增文档必须有明确职责，不复制整段原文到多个“当前说明”。历史记录里的当时路径可以保留，但应标注历史命令并提供当前入口链接；导航链接必须能打开。之前回复对 reader 计划的判断需要补充：该文件后半部已经包含实施及验证记录，问题是顶部“待执行”与后续记录没有统一状态，不能把整份文档视作未实施方案。

配置索引每行记录：配置路径、模型规模、实验用途、关键变体、适用机器/环境及对应实验文档。实验索引每行记录：研究问题、配置、入口脚本、已验证的结果路径或 commit、状态。未核实结果写“待核实”，不根据文件名推断实验完成。

## 5. 第二阶段候选结构与兼容约束

```text
scripts/
├── train.py / train.sh                 # 保持
├── test.py / test.sh                   # 保持；明确是模型评测而非单元测试
├── test_icl_baseline.py / .sh           # 保持
├── evaluation/                        # SQuAD memory 与 ICL 专用评测
├── experiments/                       # A/B、8B、队列启动
└── data/                              # 数据下载
```

现有专用 shell 路径可保留短转发入口，透传参数、环境和退出码；实际逻辑只保留一份。若兼容层会增加更多认知成本，也可暂缓全部脚本迁移，仅用索引分类。阶段二不与抽取 shell 内嵌 Python、统一 GPU 检测等行为重构混在同一改动。

已核实的迁移依赖：

- 多个 shell 用 `BASH_SOURCE[0]` 的父目录推导仓库根目录，增加目录层级会使它们定位错误，必须同步调整。
- `run_queue.sh` 调用其他启动与评测脚本；`run_on_8b.sh` 依赖指定 A/B 配置；需要检查完整调用链。
- `tests/test_ab_configs.py` 直接读取 `scripts/archive/run_ab.sh` 的源码；增加转发入口后，此项测试应读取实际实现，并单独验证转发参数和退出码。
- `tests/test_eval_logging.py` 与 `tests/test_train_schedule.py` 动态加载训练入口；保留 `scripts/train.py` 可避免无必要的兼容改动。
- 多处脚本读取 `scripts/env.local.sh`，该用户配置位置保持不变。
- `pyproject.toml` 当前打包 `src*` 与 `utils*`；本轮不改包结构。

## 6. 执行批次与验收

1. **导航批次**：新增 docs/configs/实验索引；在 README 中提供入口。验收：可由首页定位训练、恢复、评测、模型实现和实验记录；引用的当前路径存在。
2. **文档迁移批次**：按表移动文档，拆分 README，标记历史状态。验收：相对链接及 HTML 资源路径有效；实验数值、日期、commit 和证据引用未丢失；Git diff 不含运行代码和 YAML 变更。人工检查拆分内容，不能只检查 Markdown 链接。
3. **仓库卫生批次**：只增加缓存忽略规则；空 npm 清单是否删除另作明确决定。验收：`git check-ignore .hf_cache/` 生效；现有未跟踪工作文件仍保留。
4. **可选脚本迁移批次**：修改根路径定位与调用链，保留必要的旧入口。验收：`bash -n`、受影响测试、受影响脚本的 DRYRUN；先检查 DRYRUN 是否会生成配置等文件，并使用临时 WORK/LOG_DIR；不把一次目录整理变成 GPU 训练。遇到实际运行逻辑改动则另立任务。

阶段一没有必要运行 GPU 训练或编写镜像实现的测试。阶段二必须根据实际改动验证入口兼容性。每批独立 diff、独立提交（若后续要求提交）；不批量暂存未跟踪文件，不改动既有实验产物。回退按批次恢复文档或入口，不触碰运行数据。

完成标准：新读者能迅速回答“项目做什么、怎么运行、怎么评测、当前实现在哪、已有实验支持什么”；目录层级的增加只有在帮助回答这些问题时才有意义。

## 第一阶段执行记录（2026-09-18）

已完成文档迁移、首页拆分、配置/实验索引与 `.hf_cache/` 忽略。所有原有脚本、源码、测试、YAML、pyproject 和 npm 清单通过整理前后 SHA-256 对比，内容未变。文档文件链接与页内锚点经过检查；原 drawio 内容保持不变。未运行训练、评测或远端状态刷新，历史结果不作新验证结论。第二阶段脚本迁移未执行。

旧 README 的 PEFT 固定后端表述、AE 未实现表述已在当前说明中更正，原文保留在历史记录；历史 bfloat16 精度说明增加勘误。脚本和 YAML 注释中的旧文档路径保留，以满足运行文件内容不变的边界；文档导航使用新路径。
