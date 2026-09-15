def build_scheduler(optimizer, cfg, total_steps):
    from transformers import get_scheduler
    return get_scheduler(cfg.name, optimizer=optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
