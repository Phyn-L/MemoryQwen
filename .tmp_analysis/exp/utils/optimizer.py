import torch

def build_optimizer(model, cfg):
    if cfg.name.lower() != "adamw":
        raise ValueError(f"unsupported optimizer: {cfg.name}")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if any(x in name.lower() for x in ("bias", "norm", "layernorm")) else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)
