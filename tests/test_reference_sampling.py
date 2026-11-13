import pytest
torch = pytest.importorskip('torch')
datasets = pytest.importorskip('datasets')
from paperflow.statistics import prepare_reference_dataset, reference_indices


class Tokenizer:
    def __call__(self, texts):
        ids = [[int(x) for x in text.split()] for text in texts]
        return {'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]}


def test_reference_preserves_worker_remainders():
    # Four workers each receive two five-token documents. Each worker must
    # drop its own two-token remainder; a global concatenation is different.
    raw = datasets.Dataset.from_dict({
        'text': [' '.join(map(str, range(i, i + 5))) for i in range(0, 40, 5)],
        'metadata': ['unused'] * 8})
    result = prepare_reference_dataset(raw, Tokenizer(), length=4, workers=4)
    expected = [list(range(start + offset, start + offset + 4))
                for start in range(0, 40, 10) for offset in (0, 4)]
    assert result['input_ids'] == expected
    assert result['attention_mask'] == [[1] * 4 for _ in expected]


def test_reference_selection_rejects_short_corpus():
    with pytest.raises(ValueError, match='Only 2 valid blocks'):
        reference_indices(2, 3, 2023)


def test_reference_defaults_from_hydra():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
        cfg = compose(config_name='config')
        collector = instantiate(cfg.statistics)
    assert collector.dataset == 'HuggingFaceFW/fineweb'
    assert collector.subset == 'sample-10BT'
    assert collector.sampling == 'reference'
    assert (collector.documents, collector.samples, collector.sequence_length) == (500, 256, 1024)
    assert collector.device_map == 'cuda:0'
