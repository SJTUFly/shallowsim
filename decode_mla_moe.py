"""
Decode stage roofline analysis for MLA and MOE.
Extracted and refactored from shallowsim.py to accept seq_len, batch, tp, ep as parameters.
"""

import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for command-line execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class ModelArgs:
    """Model architecture arguments for DeepSeek V3."""
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class GPU_perf:
    """GPU performance parameters."""
    def __init__(self, gpu_type, sm, comm_sm, gpu_per_node,
                 fp16_flops, fp8_flops, fp4_flops,
                 mem, mem_bw, nvlink_bw, pcie_bw, discount_rate):
        self.gpu_type = gpu_type
        self.sm = sm
        self.gpu_per_node = gpu_per_node
        self.comm_sm = comm_sm
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.fp4_flops = fp4_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp4_flops(self):
        return self.fp4_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw * self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw * self.discount_rate


# ============================================================================
# Helper Functions
# ============================================================================

def get_gpu_info(filename='./device/gpu_info.csv',
                 discount_rate=0.85,
                 device_list=None,
                 decoding_mode=False,
                 print_console=False,
                 gpu_name_mapping=None):
    """
    Get GPU info from csv file.

    Args:
        filename (str): GPU performance datasheet filepath
        discount_rate (float): Estimate performance discount from Peak FLOPS and peak BW
        device_list (list): Select dedicated GPUs (None for all)
        decoding_mode (bool): Enable decoding mode to set comm_sm=0
        print_console (bool): Print result
        gpu_name_mapping (dict): Optional mapping from user GPU names to CSV GPU names

    Returns:
        dict{GPU_perf}: GPU performance dict
    """
    if device_list is None:
        device_list = []

    if gpu_name_mapping is None:
        gpu_name_mapping = {}

    gpu_dict = {}

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, trying alternative paths...")
        # Try alternative paths
        alt_paths = [
            './device/gpu_info.csv',
            '../device/gpu_info.csv',
            'device/gpu_info.csv'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                filename = alt_path
                break
        else:
            raise FileNotFoundError(f"Cannot find GPU info CSV file at {filename}")

    df = pd.read_csv(filename)

    if print_console:
        print(df.set_index('gpu_type').to_markdown())

    if decoding_mode:
        df['comm_sm'] = 0

    for _, c in df.iterrows():
        key = c['gpu_type']
        gpu = GPU_perf(
            gpu_type=c['gpu_type'],
            sm=c['sm'],
            comm_sm=c['comm_sm'],
            fp16_flops=c['fp16'],
            fp8_flops=c['fp8'],
            fp4_flops=c['fp4'],
            mem=c['mem'],
            mem_bw=c['mem_bw'],
            nvlink_bw=c['nvlink_bw'],
            pcie_bw=c['pcie_bw'],
            gpu_per_node=c['gpu_per_node'],
            discount_rate=discount_rate
        )
        if (len(device_list) == 0) or (key in device_list):
            gpu_dict[key] = gpu

        # Also add mapped names
        for user_name, csv_name in gpu_name_mapping.items():
            if key == csv_name:
                gpu_dict[user_name] = gpu

    return gpu_dict


def n_pow2_range(n: int):
    """Round up to the next power of 2."""
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n + 1
    return n


# ============================================================================
# MLA Functions
# ============================================================================

def mla_matabsob_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate=0):
    """
    Calculate MLA FLOPS with matrix absorption optimization.
    Returns total FLOPS, GEMM FLOPS, and Attention FLOPS (in GFLOPS).
    """
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_rope_up_proj = q_len * args.q_lora_rank * \
        args.n_heads * args.qk_rope_head_dim  # wq_b_rope
    q_absorb = q_len * args.n_heads * (args.q_lora_rank * args.qk_nope_head_dim  # wq_b
                                       + args.qk_nope_head_dim * args.kv_lora_rank)  # w_uk

    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)  # KV-Cache hit rate correction
    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

    # Treat as standard n_heads MQA
    mqa = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # Score_rope
                          + q_len * args.kv_lora_rank * kv_len  # Score_nope
                          + q_len * kv_len * args.kv_lora_rank)  # Score V

    attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank
    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim
    attn_sum = mqa + attn_up_proj + o_proj

    # return flops by 2* Sum(MACs)
    gemm_sum = gemm_sum * 2 / 1e9
    attn_sum = attn_sum * 2 / 1e9

    return gemm_sum + attn_sum, gemm_sum, attn_sum


def mla_weight_mem_mb(args: ModelArgs):
    """Calculate MLA weight memory usage in MB."""
    q_down_proj = args.dim * args.q_lora_rank  # wq_a
    q_up_proj = args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv
    return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj) / 1024 / 1024

def mla_KVCache_mem_mb(args: ModelArgs, batch, seq_len):
    """Calculate MLA KVCache memory usage in MB. dtype: BF16"""
    KVSize = batch * seq_len * (args.kv_lora_rank + args.qk_rope_head_dim)
    wo = args.n_heads * args.v_head_dim * args.dim  # wo
    return (KVSize * 2 + wo)/ 1024 / 1024

# ============================================================================
# MOE Functions
# ============================================================================

def moe_expert_flops(args: ModelArgs, seq_len):
    """Calculate MOE expert FLOPS in GFLOPS."""
    return 3 * seq_len * args.dim * args.moe_inter_dim * 2 / 1e9


def moe_expert_mem(args: ModelArgs):
    """Calculate MOE expert memory usage in MB."""
    return 3 * args.dim * args.moe_inter_dim / 1024 / 1024


# ============================================================================
# Decode Stage Roofline Functions
# ============================================================================

def decode_mla_roofline(seq_len, batch, tp, ep, args: ModelArgs, gpu: GPU_perf,
                       enable_gemm_fp4=False,
                       min_ar_time=0.015,
                       mla_kernel_static_time=0.05):
    """
    Calculate decode stage MLA elapsed time.

    Args:
        seq_len: Sequence length (context length)
        batch: Batch size
        tp: Tensor parallelism degree
        ep: Expert parallelism degree (not used in MLA, kept for API consistency)
        args: Model arguments
        gpu: GPU performance parameters
        enable_gemm_fp4: Enable FP4 GEMM optimization
        min_ar_time: Minimum AllReduce static latency (ms)
        mla_kernel_static_time: MLA kernel static overhead (ms)

    Returns:
        dict: {
            'gemm_fp8_flpos':GEMM FP8 Flpos,
            'attn_fp16_flpos':Attention FP16 Flpos,
            'gemm_fp8_time': GEMM FP8 computation time (ms),
            'attn_fp16_time': Attention FP16 computation time (ms),
            'load_weight_time': Weight loading time (ms),
            'load_kv_time': KV cache loading time (ms),
            'all_reduce_time': AllReduce communication time (ms),
            'total_time_no_tp': Total time without TP (ms),
            'total_time_with_tp': Total time with TP (ms)
        }
    """
    # Note: ep parameter is not used in MLA computation but kept for API consistency
    _ = ep  # Suppress unused variable warning
    # Decode: q_len=1, kv_cache_rate=1 (full cache hit)
    _, gemm_flops, attn_fp16_flops = mla_matabsob_flops(1, seq_len, args, 1)
    gemm_flops *= batch
    attn_fp16_flops *= batch
    # Compute time
    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops()
    attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops()

    # Load weight time
    mem_weight = mla_weight_mem_mb(args)
    load_weight_t = mem_weight / 1024 / gpu.get_mem_bw()
 
    # Load KV cache time
    mem_KV = mla_KVCache_mem_mb(args, batch, seq_len)
    load_kv_time = mem_KV / 1024 / gpu.get_mem_bw() * 1000

    # print('seq_len: ', seq_len,  ', ai:' , attn_fp16_flops * 1e9 / mla_KVCache_mem_mb(args, batch, seq_len) / 1024)
    # print('GB200:',2500*1e12/8000/1e9)

    # Total time
    gemm_fp8_t = max(load_weight_t, gemm_fp8_t)
    attn_fp16_t = max(load_kv_time, attn_fp16_t)
    if attn_fp16_t == load_kv_time:
        print('memory bound')
    else:
        print('compute bound')

    total_compute = gemm_fp8_t + attn_fp16_t

    # FP4 optimization
    if enable_gemm_fp4 and gpu.get_fp4_flops() > 0:
        gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()
        total_compute = max(gemm_fp4_t, load_weight_t) + attn_fp16_t

    # AllReduce communication
    ar_len = batch  # decode mode
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024 / 1024  # fp16: 2 bytes
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    # Total time with TP
    if tp == 1:
        total_time_with_tp = total_compute + mla_kernel_static_time
    else:
        total_time_with_tp = total_compute / tp + all_reduce_t + mla_kernel_static_time
        gemm_flops = gemm_flops / tp
        attn_fp16_flops = attn_fp16_flops / tp
        gemm_fp8_t = gemm_fp8_t / tp
        attn_fp16_t = attn_fp16_t / tp
        load_weight_t = load_weight_t / tp
        load_kv_time = load_kv_time / tp

    return {
        'gemm_fp8_flpos': gemm_flops,
        'attn_fp16_flpos': attn_fp16_flops,
        'gemm_fp8_time': gemm_fp8_t,
        'attn_fp16_time': attn_fp16_t,
        'mem_attn_gemm': mem_weight,
        'mem_KVCache': mem_KV,
        'load_weight_time': load_weight_t,
        'load_kv_time': load_kv_time,
        'all_reduce_time': all_reduce_t,
        'total_time_no_tp': total_compute + mla_kernel_static_time,
        'total_time_with_tp': total_time_with_tp
    }


def decode_moe_roofline(seq_len, batch, tp, ep, args: ModelArgs, gpu: GPU_perf,
                       mbs=2,
                       enable_fp4=True):
    """
    Calculate decode stage MOE expert elapsed time.

    Args:
        seq_len: Sequence length (context length, not used in current implementation)
        batch: Batch size
        tp: Tensor parallelism degree (not used in current implementation)
        ep: Expert parallelism degree (number of devices for expert parallel)
        args: Model arguments
        gpu: GPU performance parameters
        mbs: Micro batch size for pipelining
        enable_fp4: Enable FP4 optimization

    Returns:
        dict: {
            'shared_expert_time': Shared expert computation time (ms),
            'routed_expert_time': Routed expert computation time (ms),
            'gemm_group_per_device': Number of GEMM groups per device,
            'm_per_group': Average tokens per GEMM group,
            'flops_discount': FLOPS efficiency discount factor
        }
    """
    # Note: seq_len and tp parameters are not used in current MOE computation
    # but kept for API consistency
    _ = seq_len  # Suppress unused variable warning
    _ = tp  # Suppress unused variable warning

    # Calculate expert distribution
    device_num = ep
    gemm_group_per_device = math.ceil(args.n_routed_experts / device_num)

    # Memory loading overhead
    mem_moe = moe_expert_mem(args)
    load_time = mem_moe / 1024 / gpu.get_mem_bw()
    if enable_fp4 and gpu.get_fp4_flops() > 0:
        load_time = load_time / 2

    # Select GPU FLOPS
    gpu_flops = gpu.get_fp4_flops() if (enable_fp4 and gpu.get_fp4_flops() > 0) else gpu.get_fp8_flops()

    # Calculate tokens per GEMM group
    total_expert = gemm_group_per_device * device_num
    m_per_group = batch * args.n_activated_experts * device_num / total_expert / mbs

    # FLOPS discount based on group size (from profiling data)
    flops_discounts = {
        1: 0.05, 2: 0.05, 4: 0.05, 8: 0.05,
        16: 0.08, 32: 0.1, 64: 0.2, 128: 0.35,
        256: 0.4, 512: 0.6, 1024: 0.7, 2048: 0.7,
        4096: 0.7, 8192: 0.7, 16384: 0.7, 32768: 0.7, 65536: 0.7
    }

    # H20 GPU exception (different discount curve)
    if gpu.gpu_type.find('H20') != -1:
        flops_discounts = {
            1: 0.06, 2: 0.06, 4: 0.06, 8: 0.12,
            16: 0.25, 32: 0.45, 64: 0.8, 128: 0.9,
            256: 1.0, 512: 1.0, 1024: 1.0, 2048: 1.0,
            4096: 1.0, 8192: 1.0, 16384: 1.0, 32768: 1.0, 65536: 1.0
        }

    discount_factor = flops_discounts[n_pow2_range(int(m_per_group))]
    gpu_flops_effective = gpu_flops * discount_factor

    # Shared expert computation (per micro-batch)
    shared_flops = moe_expert_flops(args, batch / mbs)
    shared_time_per_mbs = shared_flops / gpu_flops_effective + load_time
    shared_time = shared_time_per_mbs * mbs

    # Routed expert computation (per micro-batch)
    num_routed_token_per_mbs = (batch / mbs) * args.n_activated_experts
    routed_flops = moe_expert_flops(args, num_routed_token_per_mbs)
    routed_time_per_mbs = routed_flops / gpu_flops_effective + load_time * gemm_group_per_device
    routed_time = routed_time_per_mbs * mbs

    return {
        'shared_expert_time': shared_time,
        'routed_expert_time': routed_time,
        'mem_moe': mem_moe,
        'gemm_group_per_device': gemm_group_per_device,
        'm_per_group': m_per_group,
        'flops_discount': discount_factor
    }


# ============================================================================
# Roofline Visualization Functions
# ============================================================================

def calculate_arithmetic_intensity(flops, memory_bytes):
    """
    Calculate arithmetic intensity (AI).

    Args:
        flops: Total floating point operations
        memory_bytes: Total memory access in bytes

    Returns:
        Arithmetic intensity (FLOP/Byte)
    """
    if memory_bytes == 0:
        return 0
    return flops / memory_bytes


def get_Theory_Data():
    """
    Generate theoretical performance data for different configurations.

    Returns:
        pd.DataFrame: Performance data with columns [GPU, batch, seq_len, TP, EP,
                      mla_total_time, moe_shared_time, moe_routed_time, total_time]
    """
    seq_len = [1024, 2048, 4096, 8192, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024]
    batch = [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 480, 512, 640, 720, 1024]
    TP = [1, 2, 4, 8, 16, 32, 64]
    EP = [1, 2, 4, 8, 16, 32, 64]
    GPUs = ['H20', 'H800', 'B200', 'GB200-NVL72']

    # GPU name mapping (user name -> CSV name)
    gpu_name_mapping = {
        'B200': 'DGX-B200'
    }

    # Get GPU info with decoding mode enabled
    gpu_dict = get_gpu_info(
        filename='./device/gpu_info.csv',
        decoding_mode=True,
        gpu_name_mapping=gpu_name_mapping
    )

    # Model arguments
    args = ModelArgs()

    # Results storage
    results = []

    # Iterate through all configurations
    total_configs = 0
    for bs in batch:
        for seq in seq_len:
            for tp in TP:
                for ep in EP:
                    if tp * ep <= 64:
                        for gpu_name in GPUs:
                            if gpu_name not in gpu_dict:
                                print(f"Warning: GPU {gpu_name} not found in GPU database, skipping...")
                                continue

                            gpu = gpu_dict[gpu_name]

                            # Call decode_mla_roofline
                            mla_result = decode_mla_roofline(
                                seq_len=seq,
                                batch=bs,
                                tp=tp,
                                ep=ep,
                                args=args,
                                gpu=gpu
                            )

                            # Call decode_moe_roofline
                            moe_result = decode_moe_roofline(
                                seq_len=seq,
                                batch=bs,
                                tp=tp,
                                ep=ep,
                                args=args,
                                gpu=gpu
                            )

                            # Store results
                            results.append({
                                'GPU': gpu_name,
                                'batch': bs,
                                'seq_len': seq,
                                'TP': tp,
                                'EP': ep,
                                'mla_result': mla_result,
                                'moe_result': moe_result,
                                'total_time_ms': mla_result['total_time_with_tp'] +
                                                 moe_result['shared_expert_time'] +
                                                 moe_result['routed_expert_time']
                            })

                            total_configs += 1
                            if total_configs % 100 == 0:
                                print(f"Processed {total_configs} configurations...")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = ['GPU', 'batch', 'seq_len', 'TP', 'EP'] + \
                   [col for col in df.columns if col not in ['GPU', 'batch', 'seq_len', 'TP', 'EP']]
    df = df[column_order]

    print(f"\nTotal configurations processed: {total_configs}")
    print(f"DataFrame shape: {df.shape}")

    return df


if __name__ == '__main__':
    print("Generating theoretical performance data...")
    print("=" * 80)

    # Generate data
    df = get_Theory_Data()

    # Print basic statistics
    print("\n" + "=" * 80)
    print("Data Summary:")
    print("=" * 80)
    print(df.head(10))
    print("\n")
    print(df.describe())

    # Save to CSV
    output_file = 'theory_performance_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")

    # Print some example queries
    print("\n" + "=" * 80)
    print("Example Analysis:")
    print("=" * 80)

    # Example 1: Best configuration for each GPU
    print("\nBest configuration (lowest total time) for each GPU:")
    best_configs = df.loc[df.groupby('GPU')['total_time_ms'].idxmin()]
    print(best_configs[['GPU', 'batch', 'seq_len', 'TP', 'EP', 'total_time_ms']])

    # Example 2: Average time by GPU
    print("\nAverage total time by GPU:")
    print(df.groupby('GPU')['total_time_ms'].mean().sort_values())
