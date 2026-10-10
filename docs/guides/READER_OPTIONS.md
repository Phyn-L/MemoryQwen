# Reader 配置

reader 在固定 memory KV 之外按问题生成 `R` 个 readout embeddings，并把它们插入 question 与 answer 之间。

```yaml
memory:
  readout_mode: memory_question
  readout_length: 8
  reader_memory_source: memory
```

`readout_mode` 可选 `memory_question`、`memory_only`、`learned_prefix`、`off`。`readout_length: 0` 总是关闭 reader。`reader_memory_source` 可选 `memory`、`memory_embedding`、`layer_mean`，只影响前两种需要 memory 的模式。

reader 的 cross attention 只处理 `M + Q` 个位置，不会增加 context writer 的长度，也不会把 question 内容写回可复用的 memory cache。
