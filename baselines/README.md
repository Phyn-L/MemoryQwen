# Memory baselines

Both baselines use the project backbone, tokenizer, data cache, training loop, loss, and
evaluation. They differ only in how the 128 memory slots are passed to the frozen decoder:
`icae` passes encoder hidden states; `500xCompressor` passes memory-slot KV cache.

```bash
conda activate shine
python -m baselines.icae.train --config baselines/icae/config.yaml
python -m baselines.500xCompressor.train --config baselines/500xCompressor/config.yaml
```

多卡训练使用项目相同的 Accelerate/DDP 入口。例如使用 4 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu --num_processes 4 \
  -m baselines.icae.train --config baselines/icae/config.yaml
```

将模块名替换为 `baselines.500xCompressor.train` 即可运行另一个 baseline。`two_stage`
会自动使用项目配置的 `find_unused_parameters=True`，单卡和多卡使用同一个 checkpoint
目录结构。

也可以使用自动检测可见 GPU 数量的启动脚本；脚本退出时会调用
`python /home/sean/burn.py`，并保留训练命令的退出码：

```bash
./baselines/train.sh icae --config baselines/icae/config.yaml --machine h200
./baselines/train.sh 500xCompressor --config baselines/500xCompressor/config.yaml --machine h200
```

Both configs select a machine through the same `machine` field as the main project. Set
`machine: h200` in the chosen baseline config, or override it at launch with `--machine h200`;
`utils/machines.py` supplies that machine's model and data roots. The default is `4090`.

Set `data.task.train_stage` in the selected config to `mix` or `two_stage`. The shared
training loop selects the learned reconstruction task embedding for `mix` and the baseline's
AE marker embedding during the reconstruction phase of `two_stage`.

Run the real-cache GPU smoke check with:

```bash
python baselines/gpu_smoke.py
```
