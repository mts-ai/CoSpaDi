# Calibration and checkpoints

- [Calibration data](https://disk.yandex.ru/d/3RwESGstkyY9lA)
- [CoSpaDi checkpoints](https://disk.yandex.ru/d/Z6k-NGVj3b_fGw)

## Calibration

| Model | Archive | Extracted folder | Base model identifier |
| :-- | :-- | :-- | :-- |
| Qwen3-0.6B | `Qwen3-0.6B-calibration.zip` | `Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |
| Qwen3-8B | `Qwen3-8B-calibration.zip` | `Qwen3-8B` | `Qwen/Qwen3-8B` |
| Llama-3.2-1B | `Llama-3.2-1B-calibration.zip` | `Llama-3.2-1B` | `unsloth/Llama-3.2-1B` |
| Llama-3-8B | `Llama-3-8B-calibration.zip` | `Llama-3-8B` | `unsloth/Llama-3-8b` |

Archives contain per-layer activation statistics. Extract the model archive and
pass its folder as `calib_root`:

```bash
unzip Qwen3-0.6B-calibration.zip -d calibration
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  calib_root="$PWD/calibration/Qwen3-0.6B" checkpoint_root=checkpoints
```

## Checkpoints

| Model | Archive | Recipe |
| :-- | :-- | :-- |
| Qwen3-0.6B | `cospadi-Qwen3-0.6B-cr0.2-fp14-fp16.zip` | `qwen3_06b_cospadi_cr02` |
| Llama-3.2-1B | `cospadi-Llama-3.2-1B-cr0.2-fp14-fp16.zip` | `llama32_1b_cospadi_cr02` |
| Llama-3-8B | `cospadi-Llama-3-8B-cr0.2-fp14-fp16.zip` | `llama3_8b_cospadi_cr02` |

FP14 denotes coefficient fake precision; dictionaries and coefficient values
are stored in FP16. Each archive contains the modeling code, safetensors and
any shard index, config/generation config, and tokenizer resources.

```bash
unzip cospadi-Qwen3-0.6B-cr0.2-fp14-fp16.zip -d checkpoints
```

Keep the extracted files together and point to the directory containing
`config.json`:

```bash
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  stages.collect=false stages.compress=false \
  checkpoint_path=/path/to/extracted/model_folder run_dir=runs/eval_qwen3_06b
```

Calibration is not needed for evaluating a checkpoint. See [USAGE.md](USAGE.md)
for model loading, calibration collection, compression and evaluation settings.
