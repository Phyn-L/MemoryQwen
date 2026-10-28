# Memory baselines

Both baselines use the project backbone, tokenizer, data cache, training loop, loss, and
evaluation. They differ only in how the 128 memory slots are passed to the frozen decoder:
`icae` passes encoder hidden states; `500xCompressor` passes memory-slot KV cache.

```bash
conda activate shine
python -m baselines.icae.train --config baselines/icae/config.yaml
python -m baselines.500xCompressor.train --config baselines/500xCompressor/config.yaml
```

Set `data.task.train_stage` in the selected config to `mix` or `two_stage`. The shared
training loop selects the learned reconstruction task embedding for `mix` and the baseline's
AE marker embedding during the reconstruction phase of `two_stage`.

Run the real-cache GPU smoke check with:

```bash
python baselines/gpu_smoke.py
```
