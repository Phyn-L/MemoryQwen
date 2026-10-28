import torch
import importlib
from utils.config import TrainConfig
from src.model import load_model
from src.data.pipeline import make_collate, make_context_dataset
from baselines.icae.model import IcaeModel

torch.cuda.set_device(2)
device = torch.device("cuda:2")
for name, path, cls in (
    ("icae", "baselines/icae/config.yaml", IcaeModel),
    ("500x", "baselines/500xCompressor/config.yaml", importlib.import_module("baselines.500xCompressor.model").Compressor500xModel),
):
    cfg = TrainConfig.from_file(path)
    cfg.data.train_num_workers = cfg.data.eval_num_workers = 0
    tokenizer, base = load_model(cfg)
    model = cls.adapt(base).to(device).train()
    dataset = make_context_dataset(cfg, "train", tokenizer, limit=1, num_workers=0)
    batch = make_collate(cfg, tokenizer, sample_qa=True, training=True)([dataset[0]])
    ids = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    embedding = model.qwen.get_input_embeddings()
    recon = {
        "context_indices": ids["recon_context_indices"],
        "context_embeds": embedding(ids["recon_target_ids"]),
        "context_mask": ids["recon_target_mask"], "labels": ids["recon_labels"],
    }
    output = model(
        embedding(ids["context_ids"]), ids["context_mask"],
        embedding(ids["qa_input_ids"]), ids["qa_input_mask"],
        embedding(ids["qa_target_ids"]), ids["qa_target_mask"], ids["qa_labels"],
        ids["qa_context_indices"], loss_config=cfg.loss, loss_only=True,
        recon_batch=recon, question_prefix_embeds=embedding(ids["qa_prefix_ids"]),
        question_prefix_mask=ids["qa_prefix_mask"],
        answer_prefix_embeds=embedding(ids["qa_answer_prefix_ids"]),
        answer_prefix_mask=ids["qa_answer_prefix_mask"],
    )
    output.supervision_losses[output.supervision_mask].mean().backward()
    prefix = model.encode_context_prefix(embedding(ids["context_ids"]), ids["context_mask"])
    memory_grad = model.memory_embeddings.grad.float().norm().item()
    print(name, tuple(prefix.memory_embedding.shape), prefix.past_key_values is not None,
          memory_grad, flush=True)
    model.eval()
    generated = model.generate_answers_with_prefix(
        prefix, ids["qa_context_indices"], ids["qa_input_ids"],
        ids["qa_input_mask"], tokenizer, max_new_tokens=2,
        answer_prefix_ids=ids["qa_answer_prefix_ids"],
        answer_prefix_mask=ids["qa_answer_prefix_mask"],
    )
    print(name, "generated", tuple(generated.shape), flush=True)
    model.zero_grad(set_to_none=True)
    empty = slice(0, 0)
    recon_only = model(
        embedding(ids["context_ids"]), ids["context_mask"],
        embedding(ids["qa_input_ids"])[empty], ids["qa_input_mask"][empty],
        embedding(ids["qa_target_ids"])[empty], ids["qa_target_mask"][empty],
        ids["qa_labels"][empty], ids["qa_context_indices"][empty],
        loss_config=cfg.loss, loss_only=True, recon_batch=recon,
    )
    recon_only.supervision_losses[recon_only.supervision_mask].mean().backward()
    marker_grad = model.ae_task_embedding.grad
    print(name, "recon-only", float(recon_only.supervision_losses.mean().detach()),
          marker_grad is not None and marker_grad.float().norm().item() > 0, flush=True)
    del model, base, output, prefix
    torch.cuda.empty_cache()
