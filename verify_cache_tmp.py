import time
from utils.config import TrainConfig
from src.model import load_model
from src.data import AggregatedQADataset, SortishSampler

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
tok, model = load_model(cfg)
names = getattr(cfg.data, "train_datasets") or cfg.data.dataset

t = time.time()
train_dataset = AggregatedQADataset(
    cfg.data.root, names, cfg.data.train_split, tok,
    cfg.data.max_context_tokens, cfg.data.train_max_samples, cfg.data.filter_long_context,
    cfg.data.filter_no_qa, allow_empty=False,
    cache_dir=cfg.checkpoint.output_dir if cfg.data.cache_dataset else None,
)
t1 = time.time() - t

t = time.time()
sampler = SortishSampler(
    train_dataset, tok, cfg.training.batch_size, cfg.data.sortish_bucket_multiplier,
    cfg.training.seed, cfg.data.use_chat_template, cfg.data.chat_template_enable_thinking,
    cfg.checkpoint.output_dir if cfg.data.cache_sortish_lengths else None,
)
t2 = time.time() - t
print(f"RESULT records={len(train_dataset)} dataset_sec={t1:.1f} sampler_sec={t2:.1f} batches={len(sampler.buckets)}")
