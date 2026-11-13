# Using CoSpaDi

## Environment

```bash
uv sync --locked --extra eval
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

The environment uses Python 3.13.5, PyTorch 2.9.1+cu128, Transformers 4.55.4,
datasets 3.6.0, Accelerate 1.7.0 and lm-eval 0.4.8. `./setup.sh` installs
the locked dependencies.

## Calibration

Download [precomputed calibration data](https://disk.yandex.ru/d/3RwESGstkyY9lA)
and pass the extracted model directory as `calib_root`.
See [ARTIFACTS.md](ARTIFACTS.md) for archive names and folder layout.

To collect new statistics:

```bash
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  stages.collect=true stages.compress=false stages.evaluate=false \
  calib_root="$PWD/calibration/Qwen3-0.6B" run_dir=runs/calibration_qwen3_06b
```

The configured source is FineWeb sample-10BT: 500 documents, 256 blocks of
1,024 tokens and seed 2023. The collector uses unnormalized FP32 Gram sums.
It profiles K/O attention and up/down MLP inputs, reusing statistics for
projections with common inputs. `statistics.arrow_file` selects a local Arrow
shard. Access to the base model and dataset is required.

## Compression

Recipes cover four models at CR 0.2, 0.3 and 0.4, using 60 K-SVD iterations and
8 power iterations. `method.ks_ratio` controls the dictionary-to-sparsity ratio.
For example:

```bash
uv run --locked --no-sync cospadi-run experiments=llama32_1b_cospadi_cr02 \
  method.ks_ratio=2 calib_root=/path/to/calibration/Llama-3.2-1B \
  checkpoint_root=checkpoints run_dir=runs/llama32_1b_cospadi_cr02
```

The Llama-3.2-1B configs use k/s=1.6; set `method.ks_ratio=2` for k/s=2.
Qwen and Llama-3-8B configs use 2. Set `stages.evaluate=false` for compression only
or `dry_run=true` to preview the commands.

Raw U/V factors remain under `checkpoint_root`. The deployable model is saved in
`transformers_fp14_float16/` beneath the factor directory; Qwen3-8B uses
`transformers_fp14_float16_balanced/`. `run_dir/results.json` records `model_path`.
Use separate checkpoint roots when changing calibration or solver settings.

### Per-layer jobs

Select a layer with `layer_idx`, write factors only and disable evaluation:

```bash
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  calib_root=/path/to/calibration/Qwen3-0.6B checkpoint_root=checkpoints \
  +method.extra_args.layer_idx=0 +method.extra_args.artifacts-only=true \
  stages.evaluate=false run_dir=runs/qwen3_06b_layer0
```

Use one visible GPU and a separate `run_dir` per layer, with a common
`checkpoint_root`. After all layers finish, run the full recipe without the
two layer flags. It loads the saved factors, exports the model and evaluates it.

## Checkpoint format

The `packed_mask_v1` format stores FP16 dictionaries, nonzero coefficient values
and packed one-bit support masks. Coefficients are truncated in BF16 to 14-bit
fake precision before FP16 storage. The allocation convention excludes masks;
the export manifest reports tensor storage including masks and buffers.
Whole-model storage also includes dense embeddings and the output head.

The `minimum` scaling policy shifts inverse powers of two between factors when
needed for FP16 range. Qwen3-8B uses `balanced` scaling to reduce underflow and
overflow. Fresh exports record these shifts in `compression_manifest.json`.

Portable forward unpacks a temporary coefficient matrix and applies two factors.
It stores no persistent fused weights or dense coefficient cache. Temporary
forward memory is additional to checkpoint storage. Load the model with
`torch_dtype=torch.float16`.

The modeling files include MACKO classes, `optimize()` and async functions.
The optimized MLP path requires compatible factorized up/gate projections and
a dense down projection. The default fully factorized model uses portable forward.

To export existing factors alongside a supporting HF checkpoint:

```bash
uv run --locked --no-sync python -m paperflow.deploy \
  --source /path/to/supporting_hf_checkpoint --factors /path/to/raw_factors \
  --output /path/to/new_output --target-cr 0.2 \
  --coeff-bits 14 --storage-dtype float16
```

Add `--factor-scaling balanced` for Qwen3-8B. Fused weights alone do not contain
the original factors. Existing output directories are not overwritten.
All experiment configs use `ksvd_deployable`, which exports packed factors.

## Evaluation

```bash
uv run --locked --no-sync cospadi-run experiments=qwen3_06b_cospadi_cr02 \
  stages.collect=false stages.compress=false \
  checkpoint_path=/path/to/extracted/model_folder run_dir=runs/eval_qwen3_06b
```

Choose the recipe matching the model. Explicitly disabling collection and
compression evaluates the checkpoint directly, without calibration data.

Evaluation uses the full zero-shot task set. Average accuracy is the unweighted
mean of MMLU, ARC-easy, ARC-challenge, HellaSwag, PIQA, LAMBADA, RACE and SciQ,
using `acc_norm` when available and `acc` otherwise. WikiText word perplexity
and LAMBADA perplexity are separate metrics.

Evaluation writes task metrics to `run_dir/evaluation/results.json`.
`evaluation=lm_eval` provides the generic harness interface for custom task settings.
