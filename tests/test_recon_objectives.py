import pytest
import torch
from src.model import MetaLoRA
from src.losses import memory_objectives, objective_total
from utils.config import MemoryConfig
from tests.test_tied_backbone import _tiny_qwen3


@pytest.mark.parametrize('embedding,prefix_ce,token,distill', [(1,0,0,0),(0,1,0,0),(1,1,1,0.3),(0,0,0,0.3)])
def test_independent_objectives_and_memory_gradients(embedding, prefix_ce, token, distill):
    torch.manual_seed(4)
    cfg = MemoryConfig(embedding_recon_weight=embedding, causal_recon_weight=prefix_ce,
                       token_recon_weight=token, distill_weight=distill,
                       token_recon_positions=3, causal_recon_positions=0, distill_positions=0)
    model = MetaLoRA(_tiny_qwen3(), target_modules=[], memory_length=4,
                    decoder_hidden_size=16, decoder_heads=4, max_context_tokens=16,
                    embedding_recon=bool(embedding), token_recon=bool(token),
                    ae_lm=bool(prefix_ce or distill))
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    prefix = model.encode_context_prefix(model.qwen.get_input_embeddings()(ids), mask)
    terms = memory_objectives(model, prefix, ids, mask, cfg)
    assert (prefix.recon is not None) == bool(embedding)
    for name, weight in [('embedding_recon',embedding), ('causal_recon',prefix_ce), ('token_recon',token), ('distill',distill)]:
        loss = terms[name + '_loss']
        assert torch.isfinite(loss)
        assert (loss.item() > 0) == bool(weight)
    objective_total(prefix.memory.sum() * 0, terms, cfg).backward()
    assert model.memory_tokens.grad.abs().sum() > 0


def test_teacher_forced_reports_distinct_context_losses():
    from src.evaluator import Evaluator
    from utils.config import TrainConfig
    cfg = TrainConfig()
    cfg.memory.embedding_recon_weight = 2.0
    cfg.memory.causal_recon_weight = 3.0
    model = MetaLoRA(_tiny_qwen3(), memory_length=4, decoder_hidden_size=16,
                    decoder_heads=4, max_context_tokens=16)
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = cfg
    # Empty shards must still reduce every loss key without dividing by zero.
    result = evaluator.teacher_forced(model, [], torch.device('cpu'))
    assert result['embedding_recon_loss'] == result['causal_recon_loss'] == 0
    assert 'reconstruction_loss' not in result and 'recon_loss' not in result
    from scripts.train import eval_log_payloads
    payload = eval_log_payloads(result)[0]
    assert 'val_teacher_forced/embedding_recon_loss' in payload
    assert 'val_teacher_forced/causal_recon_loss' in payload
