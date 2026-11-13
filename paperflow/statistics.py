"""Reference-compatible CoSpaDi activation Gram collection."""
import hashlib
import json
import pickle
import random
import time
from functools import partial
from pathlib import Path

import torch
from .utils import dump_json, torch_versions

CANONICAL_PROJECTIONS = ('self_attn.k_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.down_proj')


class GramAccumulator:
    def __init__(self, dtype='float32'):
        if dtype not in ('float32', 'float64'):
            raise ValueError('Unsupported accumulator dtype')
        self.dtype = getattr(torch, dtype)
        self.value = None
        self.tokens = 0

    @torch.no_grad()
    def add(self, activation):
        flat = activation.detach().reshape(-1, activation.shape[-1]).to(self.dtype)
        gram = (flat.T @ flat).cpu()
        if self.value is None:
            self.value = gram
        else:
            self.value += gram
        self.tokens += flat.shape[0]

    def save(self, path):
        if self.value is None:
            raise ValueError('No activations collected')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            pickle.dump(self.value, handle)


def sample_token_blocks(token_sequences, *, length, samples, seed):
    """Optional document-crop sampler; not the reference preparation path."""
    if samples <= 0 or length <= 0:
        raise ValueError('samples/length must be positive')
    rng = random.Random(seed)
    blocks = []
    for ids in token_sequences:
        if len(ids) < length:
            continue
        offset = rng.randrange(0, len(ids) - length + 1)
        blocks.append(ids[offset:offset + length])
        if len(blocks) >= samples:
            break
    if len(blocks) < samples:
        raise ValueError(f'Only {len(blocks)} valid blocks, requested {samples}')
    return blocks


def reference_tokenize(example, tokenizer, text_column='text'):
    return tokenizer(example[text_column])


def reference_group_text(examples, context_length):
    """Keep the reference map-batch boundaries, including dropped remainders."""
    concatenated = {key: sum(examples[key], []) for key in examples}
    total = len(concatenated[next(iter(concatenated))])
    total = (total // context_length) * context_length
    return {key: [values[i:i + context_length] for i in range(0, total, context_length)]
            for key, values in concatenated.items()}


def prepare_reference_dataset(dataset, tokenizer, *, length, text_column='text',
                              workers=4, batch_size=1024):
    """Match prepare_data.py's two datasets.map calls (not a global stream)."""
    tokenized = dataset.map(
        partial(reference_tokenize, tokenizer=tokenizer, text_column=text_column),
        num_proc=workers, batched=True, remove_columns=dataset.column_names)
    return tokenized.map(partial(reference_group_text, context_length=length),
                         num_proc=workers, batch_size=batch_size, batched=True)


def reference_indices(size, samples, seed):
    if samples <= 0 or samples > size:
        raise ValueError(f'Only {size} valid blocks, requested {samples}')
    # Deliberately reset the global RNG, as in build_calib_data_ksvd.py.
    torch.manual_seed(seed)
    return torch.randperm(size)[:samples]


class ActivationGramCollector:
    def __init__(self, dataset='HuggingFaceFW/fineweb', subset='sample-10BT', split='train',
                 text_column='text', samples=256, sequence_length=1024, seed=2023,
                 dtype='float32', device_map='cuda:0', modules=None, local_text_file=None,
                 trust_remote_code=False, sampling='reference', documents=500,
                 preprocessing_workers=4, preprocessing_batch_size=1024,
                 dataloader_workers=4, arrow_file=None):
        self.dataset = dataset
        self.subset = subset
        self.split = split
        self.text_column = text_column
        self.samples = samples
        self.sequence_length = sequence_length
        self.seed = seed
        self.dtype = dtype
        self.device_map = device_map
        self.modules = modules or list(CANONICAL_PROJECTIONS)
        self.local_text_file = local_text_file
        self.trust_remote_code = trust_remote_code
        self.sampling = sampling
        self.documents = documents
        self.preprocessing_workers = preprocessing_workers
        self.preprocessing_batch_size = preprocessing_batch_size
        self.dataloader_workers = dataloader_workers
        self.arrow_file = arrow_file
        if sampling not in ('reference', 'document_crop'):
            raise ValueError('sampling must be reference or document_crop')
        if sequence_length <= 0 or documents <= 0:
            raise ValueError('sequence_length/documents must be positive')

    def _dataset(self):
        from datasets import Dataset, load_dataset
        if self.local_text_file:
            raise ValueError('Use sampling=document_crop for local_text_file')
        if self.arrow_file:
            data = Dataset.from_file(str(self.arrow_file))
        else:
            data = load_dataset(self.dataset, self.subset, split=self.split,
                                trust_remote_code=self.trust_remote_code)
        if len(data) < self.documents:
            raise ValueError(f'Only {len(data)} documents, requested {self.documents}')
        return data.select(range(self.documents))

    def _texts(self):
        if self.local_text_file:
            with open(self.local_text_file, encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        yield line.strip()
        else:
            from datasets import load_dataset
            data = load_dataset(self.dataset, self.subset, split=self.split, streaming=True,
                                trust_remote_code=self.trust_remote_code)
            for record in data:
                if record[self.text_column]:
                    yield record[self.text_column]

    def run(self, ctx):
        root = Path(ctx['calib_root'])
        if ctx['dry_run']:
            print(f'DRY RUN: collect {self.sampling} {self.samples} x '
                  f'{self.sequence_length} activation tokens -> {root}')
            return {'calib_root': str(root), 'status': 'planned'}
        from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
        from torch.utils.data import DataLoader, Subset
        # Match the reference order: model, tokenizer, preparation, RNG reset, loader.
        model = AutoModelForCausalLM.from_pretrained(
            ctx['cfg'].model.name, torch_dtype=torch.bfloat16, device_map=self.device_map)
        tokenizer = AutoTokenizer.from_pretrained(ctx['cfg'].model.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        indices = None
        if self.sampling == 'reference':
            raw = self._dataset()
            train = prepare_reference_dataset(
                raw, tokenizer, length=self.sequence_length, text_column=self.text_column,
                workers=self.preprocessing_workers, batch_size=self.preprocessing_batch_size)
            indices = reference_indices(len(train), self.samples, self.seed)
            blocks = [train[int(i)]['input_ids'] for i in indices]
            loader = DataLoader(Subset(train, indices), batch_size=1, shuffle=False,
                                collate_fn=DataCollatorForLanguageModeling(
                                    tokenizer, mlm=False, return_tensors='pt'),
                                pin_memory=True, num_workers=self.dataloader_workers)
        else:
            sequences = (tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
                         for text in self._texts())
            blocks = sample_token_blocks(sequences, length=self.sequence_length,
                                         samples=self.samples, seed=self.seed)
            loader = ({'input_ids': torch.tensor([ids], dtype=torch.long)} for ids in blocks)
        token_hash = hashlib.sha256(json.dumps(blocks, separators=(',', ':')).encode()).hexdigest()
        print(f'CALIBRATION tokens_sha256={token_hash} samples={len(blocks)}', flush=True)
        model.eval()
        model.config.use_cache = False
        layers = model.model.decoder.layers if hasattr(model.model, 'decoder') else model.model.layers
        accumulators = {}
        handles = []
        start = time.time()
        try:
            for name in self.modules:
                for i, layer in enumerate(layers):
                    try:
                        submodule = layer.get_submodule(name)
                    except AttributeError as exc:
                        raise ValueError(f'Unsupported projection {name}') from exc
                    acc = GramAccumulator(self.dtype)
                    accumulators[(name, i)] = acc
                    def hook(module, inputs, output, accumulator=acc):
                        accumulator.add(inputs[0])
                    handles.append(submodule.register_forward_hook(hook))
            device = model.get_input_embeddings().weight.device
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    result = model(**{key: value.to(device) for key, value in batch.items()})
                    del result
                    if (i + 1) % 16 == 0:
                        print(f'CALIBRATION {i + 1}/{len(blocks)} '
                              f'elapsed={time.time() - start:.1f}s', flush=True)
        finally:
            for handle in handles:
                handle.remove()
        for (name, i), acc in accumulators.items():
            acc.save(root / name / f'{i}.pkl')
        info = {'dataset': self.dataset, 'subset': self.subset, 'split': self.split,
                'sampling': self.sampling, 'documents': self.documents,
                'preprocessing_workers': self.preprocessing_workers,
                'preprocessing_batch_size': self.preprocessing_batch_size,
                'arrow_file': self.arrow_file, 'samples': len(blocks),
                'sequence_length': self.sequence_length, 'seed': self.seed,
                'indices': indices.tolist() if indices is not None else None,
                'token_sha256': token_hash, 'dtype': self.dtype,
                'accumulation': 'sum of token outer products (not normalized)',
                'model': str(ctx['cfg'].model.name), 'modules': self.modules,
                'model_revision': getattr(model.config, '_commit_hash', None),
                'attention_implementation': model.config._attn_implementation,
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'environment': torch_versions(), 'collection_seconds': time.time() - start}
        dump_json(root / 'calibration_manifest.json', info)
        return {'calib_root': str(root), 'token_sha256': token_hash,
                'samples': len(blocks), 'num_matrices': len(accumulators)}
