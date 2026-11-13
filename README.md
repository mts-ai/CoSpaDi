<div align="center">

<img src="assets/cospadi.png" width="240" alt="CoSpaDi mascot">

### CoSpaDi: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning

**Denis Makhov · Dmitriy Shopkhoev · Magauiya Zhussip · Ammar Ali · Stamatios Lefkimmiatis**

**Findings of ACL: EMNLP 2026**

[Paper](https://aclanthology.org/2026.findings-emnlp.284/) · [PDF](https://aclanthology.org/2026.findings-emnlp.284.pdf) · [Usage](docs/USAGE.md) · [Checkpoints & calibration](docs/ARTIFACTS.md) · [Paper results](docs/PAPER_RESULTS.md)

<img src="assets/mws-ai.png" height="80" alt="MWS AI">
&nbsp;&nbsp;&nbsp;
<img src="assets/itmo.png" height="80" alt="ITMO University">
&nbsp;&nbsp;&nbsp;
<img src="assets/lead.png" height="80" alt="LEAD Lab">

</div>

CoSpaDi compresses transformer projection weights into a learned dictionary and
sparse coefficients. Calibration activations guide the approximation through
whitening, followed by K-SVD dictionary learning and sparse coding.

This repository provides the CoSpaDi solver, calibration tools, experiment
recipes, factorized Transformers checkpoints and zero-shot evaluation.

## Install

Use a CUDA-capable Linux machine, **Python 3.13.5** and
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mts-ai/CoSpaDi.git
cd CoSpaDi
uv sync --locked --extra eval
uv run --locked --no-sync cospadi-run --help
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

`./setup.sh` installs the same environment. The lock selects PyTorch 2.9.1+cu128,
Transformers 4.55.4 and lm-eval 0.4.8.

## Compress a model

Download the [calibration data](https://disk.yandex.ru/d/3RwESGstkyY9lA), extract
the model archive and pass its folder as `calib_root`:

```bash
unzip /path/to/Qwen3-0.6B-calibration.zip -d calibration
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  calib_root="$PWD/calibration/Qwen3-0.6B" \
  checkpoint_root="$PWD/checkpoints" run_dir="$PWD/runs/qwen3_06b_cospadi_cr02"
```

The command compresses and evaluates the model. Recipes use 60 K-SVD iterations,
8 power iterations and 14-bit coefficient fake precision. CR denotes the fraction
of decoder-core storage removed under the configured allocation convention.

| Model | Recipe prefix | CR suffixes |
| :-- | :-- | :-- |
| Qwen3-0.6B | `qwen3_06b_cospadi_` | `cr02`, `cr03`, `cr04` |
| Qwen3-8B | `qwen3_8b_cospadi_` | `cr02`, `cr03`, `cr04` |
| Llama-3.2-1B | `llama32_1b_cospadi_` | `cr02`, `cr03`, `cr04` |
| Llama-3-8B | `llama3_8b_cospadi_` | `cr02`, `cr03`, `cr04` |

Set `method.ks_ratio` to choose k/s. The Llama-3.2-1B configs use 1.6;
add `method.ks_ratio=2` to select the paper's k/s=2 setting. Other recipes use 2.
Use `dry_run=true` to preview commands and `stages.evaluate=false` for compression
only. See [usage](docs/USAGE.md) for calibration collection and per-layer jobs.

## Load a compressed model

Download a checkpoint from the [checkpoint folder](https://disk.yandex.ru/d/Z6k-NGVj3b_fGw)
and extract it. Point to the directory containing `config.json`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/path/to/extracted/model_folder"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
)
inputs = tokenizer("The purpose of model compression is", return_tensors="pt")
inputs = inputs.to(model.get_input_embeddings().weight.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0]))
```

The checkpoint stores FP16 dictionaries and sparse coefficient values with packed
one-bit masks. Coefficients have 14-bit fake precision stored in 16-bit values.
Portable forward applies the factors without a persistent fused weight copy.
Embeddings and the output head remain dense. See [artifact details](docs/ARTIFACTS.md).

## Evaluate a checkpoint

```bash
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  stages.collect=false stages.compress=false \
  checkpoint_path=/path/to/extracted/model_folder run_dir=runs/eval_qwen3_06b
```

Evaluation does not require calibration data. It runs MMLU, ARC-easy/challenge,
HellaSwag, PIQA, LAMBADA, RACE, WikiText and SciQ at zero shots without a sample
limit. See [evaluation settings](docs/USAGE.md#evaluation).

## Paper-reported results

The following CoSpaDi values are reported in
[Tables 1 and 3 of the paper](https://arxiv.org/html/2509.22075v8).
Accuracy is the eight-task average (%); lower perplexity is better.

| Model | CR | Avg. accuracy ↑ | WikiText PPL ↓ | LAMBADA PPL ↓ |
| :-- | --: | --: | --: | --: |
| Llama-3.2-1B | 0.2 | 42.7 | 64 | 35 |
| Llama-3.2-1B | 0.3 | 33.7 | 290 | 660 |
| Llama-3.2-1B | 0.4 | 27.5 | 800 | 9200 |
| Llama-3-8B | 0.2 | 61.8 | 20 | 4.3 |
| Llama-3-8B | 0.3 | 53.7 | 45 | 9.2 |
| Llama-3-8B | 0.4 | 40.7 | 180 | 120 |
| Qwen3-8B | 0.2 | 65.7 | 18 | 4.9 |
| Qwen3-8B | 0.3 | 59.97 | 23 | 6.3 |
| Qwen3-8B | 0.4 | 50.3 | 36 | 15 |

[Paper results](docs/PAPER_RESULTS.md) provides source references and a CSV.

## Citation

```bibtex
@inproceedings{makhov-etal-2026-cospadi,
    title = "{C}o{S}pa{D}i: Compressing {LLM}s via Calibration-Guided Sparse Dictionary Learning",
    author = "Makhov, Denis  and
      Shopkhoev, Dmitriy  and
      Zhussip, Magauiya  and
      Ali, Ammar  and
      Lefkimmiatis, Stamatios",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.284/",
    doi = "10.18653/2026.findings-emnlp.284",
    pages = "6005--6031",
    ISBN = "979-8-89176-462-0"
}
```

## License

Code is released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE)
for attribution. Checkpoints remain subject to their base models' licenses.
Organization logos retain their owners' trademark rights.
