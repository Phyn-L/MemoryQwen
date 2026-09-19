# Scripts 使用指南

入口按用途组织；仓库根目录保留兼容软链接，旧命令继续可用。

- `train.sh` / `train.py`：训练。
- `test.sh` / `test.py`：checkpoint 评测。
- `icl_baseline_test.sh`：Qwen ICL 基线。
- `evaluation/`：测试套件、HotpotQA/MS MARCO 评测和 ICL Python 实现。
- `experiments/`：研究消融与实验编排脚本。
- `data/`：数据下载与准备。
- `archive/`：历史专用实验脚本。
- `sync_h200.sh`：以本地为准同步 H200。

示例：

```bash
bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml
bash scripts/evaluation/test_suite.sh --machine 4090 --ckpt outputs/run/last.pt --datasets squad hotpotqa
```
