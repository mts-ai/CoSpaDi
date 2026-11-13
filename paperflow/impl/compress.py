"""CoSpaDi paper compression worker.

K-SVD, whitening, and factor saving preserve the submission implementation.
"""
import argparse
import gc
import os
import random
import time
from multiprocessing import get_context
from os import makedirs
from os.path import isfile, join
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .calibration import Calib
from .ksvd import ksvd as simp_ksvd

SEED = 42

def seed_all(seed: int = 42):
    """Initialize random seeds."""
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch cuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    # Random seed environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch versions
    
    global SEED
    SEED = seed
    
    print(f"Seeded with seed: {seed}")





def get_k_and_sparsity(cr, d1, d2, L, ks_ratio):
    vanilla = d1*d2*L
    k = (1-cr) * d1 * d2 * L / (d1 + d2*L / ks_ratio)
    sparsity = k / ks_ratio
    return int(k), int(sparsity)



module_names = [
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "self_attn.q_proj", 
    "self_attn.k_proj", 
    "self_attn.v_proj", 
    "self_attn.o_proj",
]
def frobenius_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same dimensions")
    return torch.linalg.norm(A - B, 'fro') / torch.linalg.norm(B, 'fro') 



def ksvd(w1, name="self_attn.q_proj", index=0, compression_ratio=0.2, calib_data=None, ks_ratio=2, whitening=True, svd_solver="power", n_iter=100, p_iter=8, to_horizontal=False):
    while True:
        if not isinstance(index, list):
            index = [index]
        if whitening:
            ss, inv_s = Calib.get_s_inv_s(index, name, "llama3", calib_data)
            ss = ss.cuda().double()
            inv_s = inv_s.cuda().double()
        with torch.no_grad():  # Prevent autograd from building a computation graph
            d1, d2 = w1.shape
            transpose = to_horizontal and d1 < d2
            if whitening:
                x = ss @ w1.cuda().double()
            else:
                x = w1.cuda().double()
            if transpose:
                x = x.T
                d1, d2 = d2, d1
            k_est, sparsity_est = get_k_and_sparsity(compression_ratio, d1, d2, 1, ks_ratio)
            D, C, _ = simp_ksvd(x, k_est, sparsity=sparsity_est, n_iter=n_iter, batch_size=16, svd_solver=svd_solver, power_iters=p_iter, verbose=True)
            # D, C = k_SVD((ss @ w1.float().cuda()), k_est, sparsity=sparsity_est, max_iter=10)
            if whitening:
                if transpose:
                    C = C@inv_s.T
                else:
                    D = inv_s@D 
                del ss, inv_s
        gc.collect()
        torch.cuda.empty_cache()  # Try to free CUDA memory immediately
        if not torch.isnan(D).any() and not torch.isnan(C).any():
            if transpose:
                return C.T, D.T
            return D, C


        
        

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CoSpaDi K-SVD compression')
    p.add_argument('--method', choices=('ksvd',), required=True)
    p.add_argument('--compression-ratio', type=float, required=True)
    p.add_argument('--model-name', type=str, required=True)
    p.add_argument('--calib-data', type=str, required=False)
    p.add_argument('--save-path', type=str, required=True)
    p.add_argument('--ks-ratio', type=float, default=2)
    p.add_argument('--ksvd-iterations', type=int, default=100)
    p.add_argument('--power-iterations', type=int, default=8)
    p.add_argument('--layer_idx', type=str, default='')
    p.add_argument('--artifacts-only', action='store_true',
                   help='Save reusable per-projection factors without assembling model checkpoints.')
    p.add_argument('--checkpoint-format', choices=('legacy', 'deployable_v1'), default='deployable_v1')
    p.add_argument('--coeff-bits', type=int, default=14)
    p.add_argument('--storage-dtype', choices=('float16', 'bfloat16'), default='float16')
    p.add_argument('--factor-scaling', choices=('minimum', 'balanced'), default='minimum')
    args = p.parse_args()
    if not 9 <= args.coeff_bits <= 15:
        p.error('--coeff-bits must be between 9 and 15')
    if args.checkpoint_format == 'deployable_v1' and args.layer_idx and not args.artifacts_only:
        p.error('Partial layer exports require --artifacts-only; deploy only a complete model')
    return args


def _svd_worker_wrapper(kwargs):
    # Extract seed from kwargs and pass it explicitly
    seed = kwargs.pop('seed', 42)
    return _svd_worker(**kwargs, seed=seed)

def _svd_worker(module_name: str, full_module_name: str, cr: float, index: int, w_cpu: torch.Tensor, calib_data: torch.Tensor, cfg, gpu_id: int, seed: int):

    u = v = None  # outer scope

    def compute() -> Tuple[torch.Tensor, torch.Tensor]:
        if cr < 0:
            print("!"*100)
            print(f"For module {module_name}, index: {index} detected negative CR!!!")
            print("!"*100)
        
        if cr == 0:
            u = w_cpu
            v = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
        else:
            # Seed each worker process
            seed_all(seed)  # Same seed for each process

            torch.cuda.set_device(gpu_id)
            w = w_cpu.cuda(non_blocking=True)
            # Paper K-SVD branch; numerical call retained verbatim.
            u, v = ksvd(w, name=module_name, index=index, calib_data=calib_data, compression_ratio=cr, ks_ratio=cfg.ks_ratio, n_iter=cfg.ksvd_iterations, p_iter=cfg.power_iterations)
            del w

        # Saving both u and v (both in original precision)
        print("Saving u and v...")
        makedirs(cfg.save_path, exist_ok=True)
        torch.save(u, u_full_path)
        torch.save(v, v_full_path)
        print("Done")

        return u, v

    # Check, maybe already computed
    u_full_path = join(cfg.save_path, f"{full_module_name}_u.pth")
    v_full_path = join(cfg.save_path, f"{full_module_name}_v.pth")

    if isfile(u_full_path) and isfile(v_full_path):
        try:
            print(f"Found both u anc v files for {full_module_name}, loading...")
            u = torch.load(u_full_path, map_location="cuda")
            v = torch.load(v_full_path, map_location="cuda")
            print("Done!")
        except:
            print(f"Some of u and v files was corrupted for {full_module_name}, computing...")
            u, v = compute()
    else:
        print(f"No u and v files found for {full_module_name}, computing...")
        u, v = compute()
        

    # Compute final weight
    w2 = (u.to("cuda").to(torch.bfloat16) @ v.to("cuda").to(torch.bfloat16)).cpu()
    u_cpu, v_cpu = u.cpu(), v.cpu()
    del u, v; torch.cuda.empty_cache()
    print(f"{module_name} {index}", frobenius_distance(w2, w_cpu))
    return index, w2, u_cpu, v_cpu

def auto_experiment_name(args):
    experiment_name = ''
    experiment_name += f"{args.model_name.replace('/', '_')}_"
    experiment_name += f"cr_{args.compression_ratio}_"
    experiment_name += f"{args.method}_"
    experiment_name += f"ks_{args.ks_ratio}_"
    experiment_name += f"ksvd_it_{args.ksvd_iterations}_"
    experiment_name += f"power_it_{args.power_iterations}_"
    return join(args.save_path, experiment_name[:-1])


def export_deployable(model, tokenizer, args):
    """Export factors without saving fused projection weights to the artifact."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from paperflow.compression import deployment_subdir
    from paperflow.deploy import convert_checkpoint
    output = Path(args.save_path) / deployment_subdir(
        args.coeff_bits, args.storage_dtype, args.factor_scaling)
    with TemporaryDirectory(prefix='cospadi-export-') as metadata:
        model.config.save_pretrained(metadata)
        tokenizer.save_pretrained(metadata)
        if getattr(model, 'generation_config', None) is not None:
            model.generation_config.save_pretrained(metadata)
        report = convert_checkpoint(metadata, output, args.compression_ratio,
            factors=args.save_path, state_dict=model.state_dict(), coeff_bits=args.coeff_bits,
            storage_dtype=args.storage_dtype, factor_scaling=args.factor_scaling)
    return output, report


if __name__ == "__main__":
    args = parse_args()
    seed_all()

    # Auto experiment naming
    args.save_path = auto_experiment_name(args)
    print(f"Model will be saved in {args.save_path}")

    calib_data = args.calib_data
    print(f"Model name: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"CR: {args.compression_ratio}")
    gpu_ids = list(range(torch.cuda.device_count()))
    NUM_GPUS = len(gpu_ids)
    print(f"NUM_GPUS: {NUM_GPUS}")
    num_processes = 2*NUM_GPUS
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Paper-only: fixed compression ratio applied to all projections.


    if args.layer_idx:
        if isinstance(args.layer_idx, int):
            args.layer_idx = str(args.layer_idx)
        layer_indices = [int(x) for x in args.layer_idx.split(",")]
    else:
        layer_indices = list(range(len(model.model.layers)))
    print(f"Processing layer indices: {layer_indices}")
    
    start = time.time()
    u_collection = {}
    v_collection = {}
    for module_name in module_names:
        items = []
        for i in layer_indices:
            full_module_name = f"model.layers.{i}.{module_name}"
            w_cpu = model.get_submodule(full_module_name).weight.T.detach().cpu().contiguous()
            gpu_id = gpu_ids[i % NUM_GPUS]
            items.append({
                "full_module_name": full_module_name,
                "module_name": module_name,
                "cr": args.compression_ratio,
                "index": i,
                "w_cpu": w_cpu,
                "calib_data": calib_data,
                "cfg": args,
                "gpu_id": gpu_id,
                "seed": SEED  # Pass the seed explicitly
                }
                )
        with get_context("spawn").Pool(processes=num_processes) as pool:  # 2 procs on 1 GPU
            for i, w2, u, v in tqdm(pool.imap_unordered(_svd_worker_wrapper, items), total=len(items), desc="Compressing"):
                with torch.no_grad():
                    print(f"Reloading weight for layer {i} submodule {module_name}...")
                    full_module_name_mp = f"model.layers.{i}.{module_name}"
                    model.get_submodule(full_module_name_mp).load_state_dict({"weight": w2.T})
                    print("Done!")
                    
                    print(f"Saving u and v for layer {i} submodule {module_name}...")
                    assert not full_module_name_mp in u_collection, f"For some reason {full_module_name_mp} is already in u_collection"
                    assert not full_module_name_mp in v_collection, f"For some reason {full_module_name_mp} is already in v_collection"
                    u_collection[full_module_name_mp] = u
                    v_collection[full_module_name_mp] = v
                    print("Done!")
                gc.collect(); torch.cuda.empty_cache()
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
    if args.artifacts_only:
        print(f"FACTOR_ARTIFACTS_COMPLETE: {args.save_path}", flush=True)
        raise SystemExit(0)
    if args.checkpoint_format == 'deployable_v1':
        output, report = export_deployable(model, tokenizer, args)
        print(f'DEPLOYABLE_COMPLETE: {output} actual_core_cr={report["achieved_core_cr"]:.6%}', flush=True)
        raise SystemExit(0)
    # if args.layer_idx:
    #     print("Done")
    #     exit()

    model = model.to(torch.bfloat16).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("Saving updated fused model...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Done!")

    print("Adding separate parts...")
    for module_name in module_names:
        for i in range(len(model.model.layers)):
            collection_name = f"model.layers.{i}.{module_name}"
            u_tmp = u_collection[collection_name]
            v_tmp = v_collection[collection_name]
            for postfix, value in zip(["u", "v"], [u_tmp, v_tmp]):
                setattr(model.model.layers[i].get_submodule(module_name.split(".")[0]), f"{module_name.split('.')[1]}_{postfix}", nn.Linear(in_features=value.shape[0], out_features=value.shape[1], bias=False))
                model.model.layers[i].get_submodule(module_name.split(".")[0]).get_submodule(f"{module_name.split('.')[1]}_{postfix}").load_state_dict({"weight": value.T})
    print("Done!")

    print("Saving separate model...")
    model.save_pretrained(args.save_path+"_separate")
    tokenizer.save_pretrained(args.save_path+"_separate")
    print("Done!")
