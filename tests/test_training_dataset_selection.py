from types import SimpleNamespace

import pytest

from src import pipeline


@pytest.mark.parametrize('split,selection,expected', [
    ('train', 'all', ['squad']),
    ('train', ('hotpotqa',), ('hotpotqa',)),
    ('validation', 'all', 'all'),
    ('test', ('hotpotqa',), ('hotpotqa',)),
])
def test_hotpotqa_excluded_only_from_implicit_training(tmp_path, monkeypatch, split, selection, expected):
    for name in ('hotpotqa', 'squad'):
        (tmp_path / name).mkdir()
    data = SimpleNamespace(root=str(tmp_path), dataset='all', cache_dataset=False,
                           max_context_tokens=1024, filter_long_context=True, filter_no_qa=True)
    setattr(data, f'{split}_datasets', selection)
    setattr(data, f'{split}_split', split)
    cfg = SimpleNamespace(data=data)
    monkeypatch.setattr(pipeline, 'AggregatedContextDataset', lambda root, names, *a, **kw: names)
    assert pipeline.make_context_dataset(cfg, split, None) == expected
