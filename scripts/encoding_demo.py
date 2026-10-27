"""Serve the interactive context/question encoding demo."""
from __future__ import annotations

import gc
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import TrainConfig
from utils.model_paths import relocate_saved_paths

_runtime = None


def checkpoint_paths() -> list[str]:
    paths = (path for path in ROOT.rglob("*.pt") if ".git" not in path.parts)
    return [str(path.relative_to(ROOT)) for path in sorted(
        paths, key=lambda path: path.stat().st_mtime, reverse=True,
    )]


def runtime_for(checkpoint: str):
    import torch

    path = (ROOT / checkpoint).resolve()
    if ROOT not in path.parents or not path.is_file() or path.suffix != ".pt":
        raise ValueError(f"checkpoint must be a .pt file under {ROOT}")
    saved = torch.load(path, map_location="cpu", weights_only=False).get("config")
    if not isinstance(saved, dict) or not saved:
        raise ValueError("Checkpoint has no saved config")
    # Encoding uses no dataset files; only the model path needs relocation.
    model = relocate_saved_paths({"model": saved["model"]}, None)["model"]
    cfg = TrainConfig.from_dict({
        "model": model, "memory": saved.get("memory", {}),
        "data": saved.get("data", {}),
        "evaluation": saved.get("evaluation", {}),
    })
    del saved
    from src.model import load_model
    from utils.checkpoint import CheckpointManager

    cfg.checkpoint.output_dir = str(path.parent)
    tokenizer, model = load_model(cfg)
    CheckpointManager(path.parent).load(path, model, restore_rng=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "hidden_size": int(model.qwen_hidden_size),
        "model": model,
        "device": device,
        "checkpoint": str(path),
    }


def active_runtime(checkpoint: str):
    global _runtime
    resolved = (ROOT / checkpoint).resolve()
    if _runtime is not None and _runtime["checkpoint"] != str(resolved):
        del _runtime
        _runtime = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if _runtime is None:
        _runtime = runtime_for(checkpoint)
    return _runtime


def id_rows(tokenizer, ids, mask=None) -> list[dict]:
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().tolist()
    ids = [int(value) for value in ids]
    masks = [int(value) for value in mask] if mask is not None else [1] * len(ids)
    return [
        {
            "index": index,
            "id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "text": tokenizer.decode([token_id], skip_special_tokens=False),
            "mask": masks[index],
        }
        for index, token_id in enumerate(ids)
    ]


def run_demo(payload: dict) -> dict:
    checkpoint = str(payload.get("checkpoint", ""))
    context = str(payload.get("context", ""))
    question = str(payload.get("question", ""))
    if not context.strip():
        raise ValueError("context is required")
    if payload.get("task", "qa") == "qa" and not question.strip():
        raise ValueError("question is required for QA")
    task = str(payload.get("task", "qa"))
    if task not in {"qa", "recon"}:
        raise ValueError("task must be 'qa' or 'recon'")
    import torch

    from src.evaluation.eval_metrics import qa_metrics
    from src.model import TASK_QA, TASK_RECON

    started = time.perf_counter()
    runtime = active_runtime(checkpoint)
    cfg, tokenizer = runtime["cfg"], runtime["tokenizer"]
    hidden_size, model, device = runtime["hidden_size"], runtime["model"], runtime["device"]
    from src.data.chat_prompt import split_chat_prompt
    from src.data.collate import collate_context_records
    from src.data.records import ContextRecord, ConversationMessage

    record = ContextRecord(
        context=context,
        conversations=(
            ConversationMessage("user", question),
            ConversationMessage("assistant", ""),
        ),
        independent_qa=True,
    )
    batch = collate_context_records(
        [record],
        tokenizer,
        max_context_tokens=int(cfg.data.max_context_tokens),
        max_question_tokens=int(cfg.data.max_question_tokens),
        max_answer_tokens=int(cfg.data.max_answer_tokens),
        append_eos=False,
        chat_template_enable_thinking=bool(cfg.data.chat_template_enable_thinking),
        sample_qa=False,
        question_padding_side=cfg.data.question_padding_side,
    )
    context_was_truncated = len(tokenizer(
        context,
        add_special_tokens=False,
        truncation=True,
        max_length=int(cfg.data.max_context_tokens) + 1,
    ).input_ids) > int(cfg.data.max_context_tokens)
    context_ids = batch["context_ids"].to(device)
    context_mask = batch["context_mask"].to(device)
    qa_prefix_ids = batch["qa_prefix_ids"][0]
    qa_prefix_mask = batch["qa_prefix_mask"][0]
    qa_ids = batch["qa_input_ids"][0]
    qa_mask = batch["qa_input_mask"][0]
    qa_header_ids = batch["qa_answer_prefix_ids"][0]
    qa_header_mask = batch["qa_answer_prefix_mask"][0]
    prefix_text = tokenizer.decode(
        [int(value) for value, keep in zip(qa_prefix_ids, qa_prefix_mask, strict=True) if keep],
        skip_special_tokens=False,
    )
    body_text = tokenizer.decode(
        [int(value) for value, keep in zip(qa_ids, qa_mask, strict=True) if keep],
        skip_special_tokens=False,
    )
    header_text = tokenizer.decode(
        [int(value) for value, keep in zip(qa_header_ids, qa_header_mask, strict=True) if keep],
        skip_special_tokens=False,
    )
    qa_full = prefix_text + body_text + header_text
    recon_full = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(cfg.data.chat_template_enable_thinking),
    )
    recon_user, _, recon_header = split_chat_prompt(recon_full)
    recon_user_ids = tokenizer(recon_user, add_special_tokens=False).input_ids
    recon_header_ids = tokenizer(recon_header, add_special_tokens=False).input_ids
    context_token_count = int(context_mask[0].sum().item())
    requested_budget = payload.get("max_new_tokens")
    if requested_budget in (None, ""):
        max_new_tokens = (
            context_token_count if task == "recon"
            else int(cfg.evaluation.max_new_tokens)
        )
    else:
        max_new_tokens = int(requested_budget)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be <= 4096")

    embedding = model.qwen.get_input_embeddings()
    with torch.inference_mode():
        prefix = model.encode_context_prefix(embedding(context_ids), context_mask)
        context_indices = torch.zeros(1, dtype=torch.long, device=device)
        if task == "qa":
            question_ids = batch["qa_input_ids"].to(device)
            question_mask = batch["qa_input_mask"].to(device)
            prompt_prefix_ids = batch["qa_prefix_ids"].to(device)
            prompt_prefix_mask = batch["qa_prefix_mask"].to(device)
            answer_prefix_ids = batch["qa_answer_prefix_ids"].to(device)
            answer_prefix_mask = batch["qa_answer_prefix_mask"].to(device)
            readout_mask = batch["qa_readout_mask"].to(device)
            readout_length = (
                int(cfg.memory.readout_length) if cfg.memory.readout_mode != "off" else 0
            )
            prompt_length = (
                int(cfg.memory.memory_length)
                + int(prompt_prefix_mask[0].sum().item())
                + (
                    int(question_mask[0].sum().item())
                    if not readout_length or cfg.memory.readout_keep_question_tokens
                    else 0
                )
                + readout_length
                + int(answer_prefix_mask[0].sum().item())
            )
            task_id = TASK_QA
        else:
            question_ids = torch.empty((1, 0), dtype=torch.long, device=device)
            question_mask = torch.empty((1, 0), dtype=torch.bool, device=device)
            prompt_prefix_ids = prompt_prefix_mask = None
            answer_prefix_ids = answer_prefix_mask = None
            readout_mask = None
            prompt_length = (
                int(cfg.memory.memory_length) + len(recon_user_ids) + 1 + len(recon_header_ids)
            )
            task_id = TASK_RECON
        available_tokens = model.window_limit - prompt_length
        if max_new_tokens > available_tokens:
            raise ValueError(
                f"generation budget {max_new_tokens} exceeds the {available_tokens} tokens "
                "available in this checkpoint's context window"
            )
        generated = model.generate_answers_with_prefix(
            prefix,
            context_indices,
            question_ids,
            question_mask,
            tokenizer,
            max_new_tokens=max_new_tokens,
            group_by_context=False,
            task_id=task_id,
            question_prefix_ids=prompt_prefix_ids,
            question_prefix_mask=prompt_prefix_mask,
            answer_prefix_ids=answer_prefix_ids,
            answer_prefix_mask=answer_prefix_mask,
            question_readout_mask=readout_mask,
        )
    prediction = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True).strip()
    reference = tokenizer.decode(
        context_ids[0][context_mask[0]].tolist(), skip_special_tokens=True,
    ).strip()
    task_embedding = model.recon_task_embedding.detach().float().cpu()
    return {
        "checkpoint": checkpoint,
        "context": {
            "text": context,
            "tokens": id_rows(tokenizer, context_ids[0], context_mask[0]),
            "truncated": context_was_truncated,
        },
        "question": {
            "text": question,
            "chat_template": qa_full,
            "user_prefix": prefix_text,
            "body": body_text,
            "assistant_prefix": header_text,
            "prefix_tokens": id_rows(tokenizer, qa_prefix_ids, qa_prefix_mask),
            "body_tokens": id_rows(tokenizer, qa_ids, qa_mask),
            "header_tokens": id_rows(tokenizer, qa_header_ids, qa_header_mask),
        },
        "recon": {
            "chat_template": recon_full,
            "user_tokens": id_rows(tokenizer, recon_user_ids),
            "header_tokens": id_rows(tokenizer, recon_header_ids),
            "task_embedding": {
                "positions": 1,
                "text": "learned recon_task_embedding",
                "initialization_text": "Reconstruct the context.",
                "shape": list(task_embedding.shape),
                "l2_norm": float(task_embedding.norm().item()),
                "display_values": [round(float(value), 6) for value in task_embedding[:8]],
            },
        },
        "generation": {
            "task": task,
            "prediction": prediction,
            "reference": reference if task == "recon" else None,
            "metrics": qa_metrics(prediction, reference) if task == "recon" else None,
            "token_count": int(generated.size(1)),
            "max_new_tokens": max_new_tokens,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "device": str(device),
        },
        "summary": {
            "context_tokens": context_token_count,
            "context_limit": int(cfg.data.max_context_tokens),
            "question_prefix_tokens": int(qa_prefix_mask.sum()),
            "question_body_tokens": int(qa_mask.sum()),
            "question_header_tokens": int(qa_header_mask.sum()),
            "recon_user_tokens": len(recon_user_ids),
            "recon_header_tokens": len(recon_header_ids),
            "memory_slots": int(cfg.memory.memory_length),
            "readout_mode": cfg.memory.readout_mode,
            "readout_length": int(cfg.memory.readout_length),
            "readout_keep_question_tokens": bool(cfg.memory.readout_keep_question_tokens),
            "readout_info_source": cfg.memory.question_resampler_info_source,
            "hidden_size": hidden_size,
            "context_layers": list(cfg.memory.context_layer_indices),
            "question_limit": int(cfg.data.max_question_tokens),
            "model": cfg.model.name_or_path,
            "qa_default_new_tokens": int(cfg.evaluation.max_new_tokens),
            "window_limit": int(model.window_limit),
            "recon_init_token_ids": list(model.task_init_token_ids),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if path == "/api/checkpoints":
            self._send(200, json.dumps({"checkpoints": checkpoint_paths()}).encode(), "application/json")
            return
        relative = "demo/encoding_demo.html" if path == "/" else path.lstrip("/")
        if relative != "demo/encoding_demo.html":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        file_path = (ROOT / relative).resolve()
        if ROOT not in file_path.parents or not file_path.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        self._send(200, file_path.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/encode":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            result = run_demo(json.loads(self.rfile.read(length)))
            self._send(200, json.dumps(result, ensure_ascii=False).encode(), "application/json")
        except ValueError as error:
            self._send(400, json.dumps({"error": str(error)}).encode(), "application/json")
        except Exception as error:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(error)}).encode(), "application/json")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Encoding demo: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
