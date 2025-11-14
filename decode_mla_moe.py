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
                 print_console=False):
    """
    Get GPU info from csv file.

    Args:
        filename (str): GPU performance datasheet filepath
        discount_rate (float): Estimate performance discount from Peak FLOPS and peak BW
        device_list (list): Select dedicated GPUs (None for all)
        decoding_mode (bool): Enable decoding mode to set comm_sm=0
        print_console (bool): Print result

    Returns:
        dict{GPU_perf}: GPU performance dict
    """
    if device_list is None:
        device_list = []

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
                       enable_gemm_fp4=True,
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
    load_weight_t = mla_weight_mem_mb(args) / 1024 / gpu.get_mem_bw()
 
    # Load KV cache time
    load_kv_time = mla_KVCache_mem_mb(args, batch, seq_len) / 1024 / gpu.get_mem_bw() * 1000

    # print('seq_len: ', seq_len,  ', ai:' , attn_fp16_flops * 1e9 / mla_KVCache_mem_mb(args, batch, seq_len) / 1024)
    # print('GB200:',2500*1e12/8000/1e9)

    # Total time
    gemm_fp8_t = max(load_weight_t, gemm_fp8_t)
    print('mem: ', load_kv_time, 'comp: ', attn_fp16_t)
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
        gemm_fp8_t = gemm_fp8_t / tp
        attn_fp16_t = attn_fp16_t / tp
        load_weight_t = load_weight_t / tp
        load_kv_time = load_kv_time / tp

    return {
        'gemm_fp8_time': gemm_fp8_t,
        'attn_fp16_time': attn_fp16_t,
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
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
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


def plot_roofline(gpu: GPU_perf,
                 operations=None,
                 title="Roofline Model",
                 figsize=(12, 8),
                 save_path=None,
                 show_plot=True):
    """
    Plot roofline model for GPU performance analysis.

    Args:
        gpu: GPU_perf object with performance parameters
        operations: List of dictionaries containing operation info:
            [
                {
                    'name': 'Operation Name',
                    'theoretical_flops': Theoretical FLOPS (in GFLOPS),
                    'achieved_performance': Achieved performance (in TFLOPS),
                    'memory': Memory access (in GB),
                    'color': 'color' (optional),
                    'marker': 'marker style' (optional)
                },
                ...
            ]
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    if operations is None:
        operations = []

    fig, ax = plt.subplots(figsize=figsize)

    # Get GPU peak performance (in TFLOPS)
    peak_fp16 = gpu.get_fp16_flops()  # TFLOPS
    peak_fp8 = gpu.get_fp8_flops()    # TFLOPS
    peak_fp4 = gpu.get_fp4_flops()    # TFLOPS
    mem_bw = gpu.get_mem_bw()         # GB/s

    # Calculate ridge points (where memory bandwidth meets compute ceiling)
    ridge_fp16 = peak_fp16 * 1000 / mem_bw  # FLOP/Byte
    ridge_fp8 = peak_fp8 * 1000 / mem_bw

    # Generate arithmetic intensity range (FLOP/Byte) - extend to 10000
    ai_range = np.logspace(-2, 4, 600)  # 0.01 to 10000 FLOP/Byte

    # Memory bandwidth bound (convert to TFLOPS)
    mem_bound = ai_range * mem_bw / 1000  # TFLOPS

    # Compute bound lines (in TFLOPS)
    compute_fp16 = np.ones_like(ai_range) * peak_fp16
    compute_fp8 = np.ones_like(ai_range) * peak_fp8

    # Plot roofline curves
    ax.loglog(ai_range, np.minimum(mem_bound, compute_fp16),
             'b-', linewidth=2.5, label=f'FP16 Peak ({peak_fp16:.0f} TFLOPS)')
    ax.loglog(ai_range, np.minimum(mem_bound, compute_fp8),
             'g-', linewidth=2.5, label=f'FP8 Peak ({peak_fp8:.0f} TFLOPS)')

    if peak_fp4 > 0:
        ridge_fp4 = peak_fp4 * 1000 / mem_bw
        compute_fp4 = np.ones_like(ai_range) * peak_fp4
        ax.loglog(ai_range, np.minimum(mem_bound, compute_fp4),
                 'r-', linewidth=2.5, label=f'FP4 Peak ({peak_fp4:.0f} TFLOPS)')

    # Plot ridge points
    ax.plot(ridge_fp16, peak_fp16, 'bo', markersize=8, label=f'FP16 Ridge ({ridge_fp16:.2f} FLOP/B)')
    ax.plot(ridge_fp8, peak_fp8, 'go', markersize=8, label=f'FP8 Ridge ({ridge_fp8:.2f} FLOP/B)')
    if peak_fp4 > 0:
        ax.plot(ridge_fp4, peak_fp4, 'ro', markersize=8, label=f'FP4 Ridge ({ridge_fp4:.2f} FLOP/B)')

    # Add memory bandwidth line (already shown as part of roofline)
    # Just add a label annotation for clarity
    mem_bound_line_x = ai_range[ai_range < ridge_fp8]
    mem_bound_line_y = mem_bound[ai_range < ridge_fp8]
    ax.plot(mem_bound_line_x, mem_bound_line_y, 'gray', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Memory BW Bound ({mem_bw:.0f} GB/s)')


    # Plot operation points
    if operations:
        # Collect all AI values to intelligently position annotations
        op_points = []
        for op in operations:
            name = op.get('name', 'Unknown')
            theoretical_flops = op.get('theoretical_flops', 0)  # GFLOPS (theoretical)
            achieved_perf = op.get('achieved_performance', 0)  # TFLOPS (achieved)
            memory = op.get('memory', 0)  # GB
            color = op.get('color', 'red')
            marker = op.get('marker', 'o')

            if memory > 0 and theoretical_flops > 0 and achieved_perf > 0:  # Ensure valid values
                # AI = Theoretical FLOPS / Memory bytes
                ai = calculate_arithmetic_intensity(theoretical_flops * 1e9, memory * 1e9)  # FLOP/Byte
                # Y-axis = Achieved performance
                op_points.append((ai, achieved_perf, name, color, marker))

        # Sort by AI to position annotations smartly
        op_points.sort(key=lambda x: x[0])

        # Plot points and add smart annotations
        for i, (ai, perf, name, color, marker) in enumerate(op_points):
            ax.plot(ai, perf, marker, color=color, markersize=12,
                   label=name, markeredgecolor='black', markeredgewidth=1.5, zorder=10)

            # Smart annotation positioning to avoid overlap
            # Alternate between top-right, top-left, bottom-right, bottom-left
            positions = [
                (15, 15),   # top-right
                (-80, 15),  # top-left
                (15, -25),  # bottom-right
                (-80, -25), # bottom-left
            ]
            xytext = positions[i % len(positions)]

            # Add annotation without blocking the point
            ax.annotate(name, xy=(ai, perf), xytext=xytext,
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.4', fc=color, alpha=0.5, edgecolor='black', linewidth=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                     color='black', linewidth=0.8),
                       zorder=5)

    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n{gpu.gpu_type} GPU', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Set reasonable axis limits
    ax.set_xlim([0.01, 10000])
    ax.set_ylim([0.001, max(peak_fp8 * 1.5, 10)])

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Roofline plot saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()

    return fig, ax


def plot_roofline_with_lines(gpu: GPU_perf,
                              operations=None,
                              boundary_annotations=None,
                              ridge_annotations=None,
                              title="Roofline Model",
                              figsize=(12, 8),
                              save_path=None,
                              show_plot=True):
    """
    Plot roofline model with operator types connected by lines.

    Args:
        gpu: GPU_perf object with performance parameters
        operations: List of dictionaries containing operation info with 'ai' field
        boundary_annotations: List of tuples (ai, perf, batch_size, seq_len) for boundary markers
        ridge_annotations: List of tuples (ai, perf, batch_size, operator_type) for ridge point markers
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    if operations is None:
        operations = []
    if boundary_annotations is None:
        boundary_annotations = []
    if ridge_annotations is None:
        ridge_annotations = []

    fig, ax = plt.subplots(figsize=figsize)

    # Get GPU peak performance (in TFLOPS)
    peak_fp16 = gpu.get_fp16_flops()
    peak_fp8 = gpu.get_fp8_flops()
    peak_fp4 = gpu.get_fp4_flops()
    mem_bw = gpu.get_mem_bw()

    # Calculate ridge points
    ridge_fp16 = peak_fp16 * 1000 / mem_bw
    ridge_fp8 = peak_fp8 * 1000 / mem_bw

    # Generate arithmetic intensity range
    ai_range = np.logspace(-2, 4, 600)

    # Memory bandwidth bound (convert to TFLOPS)
    mem_bound = ai_range * mem_bw / 1000

    # Compute bound lines (in TFLOPS)
    compute_fp16 = np.ones_like(ai_range) * peak_fp16
    compute_fp8 = np.ones_like(ai_range) * peak_fp8

    # Plot roofline curves
    ax.loglog(ai_range, np.minimum(mem_bound, compute_fp16),
             'b-', linewidth=2.5, label=f'FP16 Peak ({peak_fp16:.0f} TFLOPS)')
    ax.loglog(ai_range, np.minimum(mem_bound, compute_fp8),
             'g-', linewidth=2.5, label=f'FP8 Peak ({peak_fp8:.0f} TFLOPS)')

    if peak_fp4 > 0:
        ridge_fp4 = peak_fp4 * 1000 / mem_bw
        compute_fp4 = np.ones_like(ai_range) * peak_fp4
        ax.loglog(ai_range, np.minimum(mem_bound, compute_fp4),
                 'r-', linewidth=2.5, label=f'FP4 Peak ({peak_fp4:.0f} TFLOPS)')

    # Plot ridge points
    ax.plot(ridge_fp16, peak_fp16, 'bo', markersize=8, label=f'FP16 Ridge ({ridge_fp16:.2f} FLOP/B)')
    ax.plot(ridge_fp8, peak_fp8, 'go', markersize=8, label=f'FP8 Ridge ({ridge_fp8:.2f} FLOP/B)')
    if peak_fp4 > 0:
        ax.plot(ridge_fp4, peak_fp4, 'ro', markersize=8, label=f'FP4 Ridge ({ridge_fp4:.2f} FLOP/B)')

    # Draw vertical line at FP8 ridge point (compute/memory bound boundary)
    ax.axvline(x=ridge_fp8, color='purple', linestyle='--', linewidth=2.5,
               alpha=0.7, label=f'Boundary (AI={ridge_fp8:.1f})', zorder=3)

    # Group operations by type
    if operations:
        op_types = {}
        for op in operations:
            op_type = op.get('operator_type', 'Unknown')
            if op_type not in op_types:
                op_types[op_type] = []

            ai = op.get('ai', 0)
            perf = op.get('achieved_performance', 0)
            color = op.get('color', 'red')
            marker = op.get('marker', 'o')

            op_types[op_type].append((ai, perf, color, marker))

        # Plot each operator type with connected lines
        arrow_added = {'MLA GEMM': False, 'MOE': False, 'MLA Attention': False}

        for op_type, points in op_types.items():
            if not points:
                continue

            # Sort points by AI for proper line connection
            points.sort(key=lambda x: x[0])

            ais = [p[0] for p in points]
            perfs = [p[1] for p in points]
            color = points[0][2]
            marker = points[0][3]

            # Determine the base operator type (remove batch/seq info)
            base_op_type = op_type.split('(')[0].strip()

            # Plot line connecting all points
            ax.plot(ais, perfs, '-', color=color, linewidth=1.5, alpha=0.6)

            # Plot points
            # For legend: only show base operator types, not individual curves
            if not arrow_added.get(base_op_type, False):
                ax.plot(ais, perfs, marker, color=color, markersize=8,
                       label=f'{base_op_type}', markeredgecolor='black', markeredgewidth=1,
                       zorder=10)
                arrow_added[base_op_type] = True
            else:
                ax.plot(ais, perfs, marker, color=color, markersize=8,
                       markeredgecolor='black', markeredgewidth=1,
                       zorder=10)

            # Add arrow only if points are not overlapping
            # Check if there are distinct AI values (not all the same)
            unique_ais = len(set([round(ai, 2) for ai in ais]))

            if unique_ais >= 2 and len(ais) >= 2:
                # Find non-overlapping segment for arrow
                # Add arrow at 70% position along the curve
                arrow_idx = int(len(ais) * 0.7)
                if arrow_idx >= len(ais) - 1:
                    arrow_idx = len(ais) - 2

                # Arrow from point arrow_idx to arrow_idx+1
                x_start, y_start = ais[arrow_idx], perfs[arrow_idx]
                x_end, y_end = ais[arrow_idx + 1], perfs[arrow_idx + 1]

                # Only add arrow if points are sufficiently far apart
                if abs(x_end - x_start) / x_start > 0.01 or abs(y_end - y_start) / max(y_start, 0.001) > 0.01:
                    # Add arrow annotation
                    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                             mutation_scale=20, alpha=0.8),
                               zorder=5)

    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n{gpu.gpu_type} GPU', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Add text box explaining arrows
    arrow_text = "Arrows indicate:\n• MLA GEMM: batch ↑\n• MLA Attention: seq_len ↓\n• MOE: batch ↑"
    ax.text(0.98, 0.02, arrow_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
            zorder=15)

    # Add boundary annotations (compute/memory bound transition points)
    if boundary_annotations:
        for ai, perf, batch, seq_len in boundary_annotations:
            # Plot boundary marker
            ax.plot(ai, perf, 'o', color='red', markersize=10,
                   markeredgecolor='black', markeredgewidth=2, zorder=20)

            # Add annotation with batch size
            annotation_text = f'BS={batch}'
            ax.annotate(annotation_text, xy=(ai, perf), xytext=(10, 10),
                       textcoords='offset points', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', fc='red', alpha=0.7, edgecolor='black', linewidth=1.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     color='red', linewidth=1.5),
                       zorder=25)

    # Add ridge annotations (batch sizes near compute/memory boundary)
    if ridge_annotations:
        # Offset annotations vertically to avoid overlap
        y_offsets = [15, -30]  # Alternate up and down
        for idx, (ai, perf, batch, op_type) in enumerate(ridge_annotations):
            # Plot marker with operator-specific color
            marker_color = 'blue' if op_type == 'MLA GEMM' else 'darkgreen'
            ax.plot(ai, perf, 'o', color=marker_color, markersize=10,
                   markeredgecolor='purple', markeredgewidth=3, zorder=20)

            # Add annotation with batch size
            annotation_text = f'{op_type}\nBS={batch}'
            y_offset = y_offsets[idx % len(y_offsets)]
            ax.annotate(annotation_text, xy=(ai, perf), xytext=(0, y_offset),
                       textcoords='offset points', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', fc=marker_color, alpha=0.7,
                                edgecolor='purple', linewidth=2),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     color='purple', linewidth=1.5),
                       zorder=25, ha='center')

    # Set reasonable axis limits
    ax.set_xlim([0.01, 10000])
    ax.set_ylim([0.001, max(peak_fp8 * 1.5, 10)])

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Roofline plot saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()

    return fig, ax


def visualize_decode_operations(seq_len, batch, tp, ep,
                                args: ModelArgs, gpu: GPU_perf,
                                title="Decode Stage Roofline Analysis",
                                save_path=None,
                                show_plot=True):
    """
    Visualize MLA and MOE operations on roofline plot for decode stage.
    Y-axis shows actual achieved performance (TFLOPS) = FLOPS / time.

    Args:
        seq_len: Sequence length
        batch: Batch size
        tp: Tensor parallelism degree
        ep: Expert parallelism degree
        args: Model arguments
        gpu: GPU performance parameters
        title: Plot title
        save_path: Path to save the figure
        show_plot: Whether to display the plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    # Calculate MLA performance metrics
    mla_result = decode_mla_roofline(seq_len, batch, tp, ep, args, gpu)
    moe_result = decode_moe_roofline(seq_len, batch, tp, ep, args, gpu)

    # MLA FLOPS calculations
    mla_total_flops, mla_gemm_flops, mla_attn_flops = mla_matabsob_flops(1, seq_len, args, 1)
    mla_total_flops *= batch
    mla_gemm_flops *= batch
    mla_attn_flops *= batch

    # MLA memory access (parameters + KV cache)
    # Memory access should correspond to actual load operations
    mla_params_mb = mla_weight_mem_mb(args)  # Weight parameters for GEMM
    kv_cache_size_tokens = mla_KVCache_mem_mb(args, batch, seq_len)
    kv_cache_size_mb = kv_cache_size_tokens / 1024 / 1024  # MB

    # Per-GPU memory access (divided by TP)
    # For MLA GEMM: load weights (FP8, 1 byte per param)
    mla_gemm_memory_gb = (mla_params_mb * 1) / 1024 / tp  # FP8: 1 byte

    # For MLA Attention: load KV cache (FP16, 2 bytes per element)
    mla_attn_memory_gb = (kv_cache_size_mb * 2) / 1024 / tp  # FP16: 2 bytes

    # For MLA Total: sum of GEMM + Attention memory
    mla_total_memory_gb = mla_gemm_memory_gb + mla_attn_memory_gb

    # MOE FLOPS calculations
    shared_flops = moe_expert_flops(args, batch)
    routed_token_count = batch * args.n_activated_experts
    routed_flops = moe_expert_flops(args, routed_token_count)

    # MOE memory access
    expert_params_mb = moe_expert_mem(args)
    gemm_group_per_device = math.ceil(args.n_routed_experts / ep)
    shared_memory_gb = expert_params_mb / 1024
    routed_memory_gb = expert_params_mb * gemm_group_per_device / 1024

    # Calculate actual performance (TFLOPS) per GPU
    # Time returned from decode functions is in SECONDS (not ms despite the printout)
    # Performance = (FLOPS_per_GPU) / (time_in_seconds) / 1000 to convert to TFLOPS
    #             = (total_FLOPS / tp) / time_seconds / 1000

    # For MLA with TP, each GPU handles batch/tp tokens
    mla_total_perf = (mla_total_flops / tp) / mla_result['total_time_with_tp'] / 1000  # TFLOPS
    mla_gemm_perf = (mla_gemm_flops / tp) / mla_result['gemm_fp8_time'] / 1000  # TFLOPS
    mla_attn_perf = (mla_attn_flops / tp) / mla_result['attn_fp16_time'] / 1000  # TFLOPS

    # For MOE, workload is already per-device
    moe_shared_perf = shared_flops / moe_result['shared_expert_time'] / 1000  # TFLOPS
    moe_routed_perf = routed_flops / moe_result['routed_expert_time'] / 1000  # TFLOPS

    # Prepare operations list with actual achieved performance (per GPU, in TFLOPS)
    # Note: AI should be calculated using THEORETICAL FLOPS, not achieved performance
    # Memory corresponds to actual load operations (load_weight, load_kv)
    operations = [
        {
            'name': f'MLA Total\n(BS={batch}, TP={tp})',
            'theoretical_flops': mla_total_flops / tp,  # Theoretical GFLOPS per GPU
            'achieved_performance': mla_total_perf,  # Actual achieved TFLOPS per GPU
            'memory': mla_total_memory_gb,  # load_weight + load_kv
            'color': 'blue',
            'marker': 's'
        },
        {
            'name': f'MLA GEMM\n(FP8)',
            'theoretical_flops': mla_gemm_flops / tp,  # Theoretical GFLOPS per GPU
            'achieved_performance': mla_gemm_perf,  # Actual achieved TFLOPS per GPU
            'memory': mla_gemm_memory_gb,  # load_weight only
            'color': 'cyan',
            'marker': '^'
        },
        {
            'name': f'MLA Attn\n(FP16)',
            'theoretical_flops': mla_attn_flops / tp,  # Theoretical GFLOPS per GPU
            'achieved_performance': mla_attn_perf,  # Actual achieved TFLOPS per GPU
            'memory': mla_attn_memory_gb,  # load_kv only
            'color': 'lightblue',
            'marker': 'v'
        },
        {
            'name': f'MOE Shared\n(BS={batch})',
            'theoretical_flops': shared_flops,  # Theoretical GFLOPS per GPU
            'achieved_performance': moe_shared_perf,  # Actual achieved TFLOPS per GPU
            'memory': shared_memory_gb * 1,  # fp8: 1 byte (or fp4: 0.5)
            'color': 'green',
            'marker': 'o'
        },
        {
            'name': f'MOE Routed\n(EP={ep})',
            'theoretical_flops': routed_flops,  # Theoretical GFLOPS per GPU
            'achieved_performance': moe_routed_perf,  # Actual achieved TFLOPS per GPU
            'memory': routed_memory_gb * 1,
            'color': 'darkgreen',
            'marker': 'D'
        }
    ]

    # Plot roofline
    fig, ax = plot_roofline(
        gpu=gpu,
        operations=operations,
        title=f"{title}\nseq_len={seq_len}, batch={batch}, TP={tp}, EP={ep}",
        save_path=save_path,
        show_plot=show_plot
    )

    return fig, ax


def visualize_decode_sweeps(seq_lens, batch_sizes, tp, ep,
                            args: ModelArgs, gpu: GPU_perf,
                            title="Decode Stage Roofline Sweep",
                            save_path=None,
                            show_plot=True):
    """
    Visualize MLA and MOE operations with multiple seq_len and batch_size configurations
    on a single roofline plot. Points are connected by lines for each operator type.

    MLA is split into GEMM and Attention components:
    - MLA GEMM: Only varies with batch (decode q_len=1, weight loading is fixed)
    - MLA Attention: Varies with both seq_len and batch (KV cache access)
    - MOE: Only varies with batch (seq_len is not used in MOE computation)

    Args:
        seq_lens: List of sequence lengths to sweep
        batch_sizes: List of batch sizes to sweep
        tp: Tensor parallelism degree (fixed)
        ep: Expert parallelism degree (fixed)
        args: Model arguments
        gpu: GPU performance parameters
        title: Plot title
        save_path: Path to save the figure
        show_plot: Whether to display the plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    # Collect data for MLA GEMM, MLA Attention, and MOE separately
    mla_gemm_data = []  # (AI, performance, batch) - only varies with batch
    mla_attn_data_by_batch = {}  # batch -> [(AI, performance, seq_len), ...]
    moe_data = []  # (AI, performance, batch) - only varies with batch

    # First, collect MLA GEMM and MOE data (only varies with batch)
    # Use the first seq_len for these since they don't depend on seq_len
    reference_seq_len = seq_lens[0]
    for batch in batch_sizes:
        # Calculate MLA and MOE results
        mla_result = decode_mla_roofline(reference_seq_len, batch, tp, ep, args, gpu)
        moe_result = decode_moe_roofline(reference_seq_len, batch, tp, ep, args, gpu)

        # MLA GEMM FLOPS (only the GEMM part)
        _, mla_gemm_flops, _ = mla_matabsob_flops(1, reference_seq_len, args, 1)
        mla_gemm_flops *= batch

        # MLA GEMM memory (only weight loading)
        mla_params_mb = mla_weight_mem_mb(args)
        mla_gemm_memory_gb = (mla_params_mb * 1) / 1024 / tp  # FP8: 1 byte

        # MLA GEMM performance
        mla_gemm_perf = (mla_gemm_flops / tp) / mla_result['gemm_fp8_time'] / 1000  # TFLOPS

        # MLA GEMM AI
        mla_gemm_ai = calculate_arithmetic_intensity(mla_gemm_flops / tp * 1e9, mla_gemm_memory_gb * 1e9)

        mla_gemm_data.append((mla_gemm_ai, mla_gemm_perf, batch))

        # MOE calculations
        routed_token_count = batch * args.n_activated_experts
        routed_flops = moe_expert_flops(args, routed_token_count)

        expert_params_mb = moe_expert_mem(args)
        gemm_group_per_device = math.ceil(args.n_routed_experts / ep)
        routed_memory_gb = expert_params_mb * gemm_group_per_device / 1024

        # MOE performance
        moe_routed_perf = routed_flops / moe_result['routed_expert_time'] / 1000  # TFLOPS

        # MOE AI
        moe_ai = calculate_arithmetic_intensity(routed_flops * 1e9, routed_memory_gb * 1e9)

        moe_data.append((moe_ai, moe_routed_perf, batch))

    # Now collect MLA Attention data (varies with both seq_len and batch)
    # Organize by batch so we can connect different seq_len points
    # Also track boundary points (where compute/memory bound transition occurs)
    boundary_annotations = []  # List of (ai, perf, batch_size, seq_len) for annotation

    for batch in batch_sizes:
        mla_attn_data_by_batch[batch] = []
        for seq_len in seq_lens:
            # Calculate MLA results for this configuration
            mla_result = decode_mla_roofline(seq_len, batch, tp, ep, args, gpu)

            # MLA Attention FLOPS (only the attention part)
            _, _, mla_attn_flops = mla_matabsob_flops(1, seq_len, args, 1)
            mla_attn_flops *= batch

            # MLA Attention memory (only KV cache loading)
            kv_cache_size_tokens = mla_KVCache_mem_mb(args, batch, seq_len)
            kv_cache_size_mb = kv_cache_size_tokens / 1024 / 1024
            mla_attn_memory_gb = (kv_cache_size_mb * 2) / 1024 / tp  # FP16: 2 bytes

            # MLA Attention performance
            mla_attn_perf = (mla_attn_flops / tp) / mla_result['attn_fp16_time'] / 1000  # TFLOPS

            # MLA Attention AI
            mla_attn_ai = calculate_arithmetic_intensity(mla_attn_flops / tp * 1e9, mla_attn_memory_gb * 1e9)

            mla_attn_data_by_batch[batch].append((mla_attn_ai, mla_attn_perf, seq_len))

    # Find boundary points for each seq_len (where compute/memory bound transition occurs)
    for seq_len in seq_lens:
        is_memory_bound_prev = None
        for batch in batch_sizes:
            # Recalculate to check if memory or compute bound
            mla_result = decode_mla_roofline(seq_len, batch, tp, ep, args, gpu)

            # Check if memory bound (load_kv_time >= attn_fp16_time)
            is_memory_bound = (mla_result['load_kv_time'] >= mla_result['attn_fp16_time'])

            # Detect transition
            if is_memory_bound_prev is not None and is_memory_bound != is_memory_bound_prev:
                # Boundary detected! Mark this point
                # Get the data for this configuration
                _, _, mla_attn_flops = mla_matabsob_flops(1, seq_len, args, 1)
                mla_attn_flops *= batch
                kv_cache_size_tokens = mla_KVCache_mem_mb(args, batch, seq_len)
                kv_cache_size_mb = kv_cache_size_tokens / 1024 / 1024
                mla_attn_memory_gb = (kv_cache_size_mb * 2) / 1024 / tp
                mla_attn_perf = (mla_attn_flops / tp) / mla_result['attn_fp16_time'] / 1000
                mla_attn_ai = calculate_arithmetic_intensity(mla_attn_flops / tp * 1e9, mla_attn_memory_gb * 1e9)

                boundary_annotations.append((mla_attn_ai, mla_attn_perf, batch, seq_len))

            is_memory_bound_prev = is_memory_bound

    # Calculate ridge point (compute/memory bound boundary for FP8)
    peak_fp8 = gpu.get_fp8_flops()
    mem_bw = gpu.get_mem_bw()
    ridge_fp8 = peak_fp8 * 1000 / mem_bw  # FLOP/Byte

    # Find MLA GEMM and MOE points closest to ridge_fp8
    ridge_annotations = []  # List of (ai, perf, batch_size, operator_type)

    # Find closest MLA GEMM point
    if mla_gemm_data:
        closest_gemm = min(mla_gemm_data, key=lambda x: abs(x[0] - ridge_fp8))
        ai, perf, batch = closest_gemm
        # Only annotate if reasonably close (within 50% of ridge)
        if abs(ai - ridge_fp8) / ridge_fp8 < 0.5:
            ridge_annotations.append((ai, perf, batch, 'MLA GEMM'))

    # Find closest MOE point
    if moe_data:
        closest_moe = min(moe_data, key=lambda x: abs(x[0] - ridge_fp8))
        ai, perf, batch = closest_moe
        # Only annotate if reasonably close (within 50% of ridge)
        if abs(ai - ridge_fp8) / ridge_fp8 < 0.5:
            ridge_annotations.append((ai, perf, batch, 'MOE'))

    # Prepare operations for plotting (without individual labels)
    operations = []

    # Add MLA GEMM points (only varies with batch)
    for ai, perf, batch in mla_gemm_data:
        operations.append({
            'theoretical_flops': 1,  # Dummy value, not used for plotting
            'achieved_performance': perf,
            'memory': 1,  # Dummy value
            'ai': ai,  # Pass AI directly
            'color': 'blue',
            'marker': 's',
            'operator_type': 'MLA GEMM'
        })

    # Add MLA Attention points (each batch gets a separate curve connecting different seq_lens)
    for batch in batch_sizes:
        points = mla_attn_data_by_batch[batch]
        # Sort by seq_len in DESCENDING order (so arrow points toward decreasing seq_len)
        points.sort(key=lambda x: x[2], reverse=True)  # Sort by seq_len descending

        for ai, perf, seq_len in points:
            operations.append({
                'theoretical_flops': 1,
                'achieved_performance': perf,
                'memory': 1,
                'ai': ai,
                'color': 'cyan',
                'marker': '^',
                'operator_type': f'MLA Attention (batch={batch})'
            })

    # Add MOE points (only varies with batch)
    for ai, perf, batch in moe_data:
        operations.append({
            'theoretical_flops': 1,
            'achieved_performance': perf,
            'memory': 1,
            'ai': ai,
            'color': 'darkgreen',
            'marker': 'D',
            'operator_type': 'MOE'
        })

    # Plot roofline with connected lines
    fig, ax = plot_roofline_with_lines(
        gpu=gpu,
        operations=operations,
        boundary_annotations=boundary_annotations,
        ridge_annotations=ridge_annotations,
        title=f"{title}\nTP={tp}, EP={ep}",
        figsize=(14, 10),
        save_path=save_path,
        show_plot=show_plot
    )

    return fig, ax


def batch_sweep_analysis(seq_len, tp, ep, batch_sizes, args: ModelArgs, gpu: GPU_perf):
    """
    Perform batch size sweep analysis with fixed seq_len, tp, ep.

    Args:
        seq_len: Fixed sequence length
        tp: Fixed tensor parallelism degree
        ep: Fixed expert parallelism degree
        batch_sizes: List of batch sizes to sweep
        args: Model arguments
        gpu: GPU performance parameters

    Returns:
        Dictionary containing batch sizes and corresponding metrics
    """
    results = {
        'batch_sizes': [],
        'total_time': [],
        'gemm_time': [],
        'attn_time': [],
        'load_weight_time': [],
        'load_kv_time': [],
        'all_reduce_time': []
    }

    print(f"Running batch size sweep analysis on {gpu.gpu_type}")
    print(f"Fixed parameters: seq_len={seq_len}, tp={tp}, ep={ep}")
    print(f"Batch sizes: {batch_sizes}")
    print("=" * 80)

    # Run analysis for each batch size
    for batch in batch_sizes:
        print(f"\nAnalyzing batch size: {batch}")

        # Call decode_mla_roofline
        mla_result = decode_mla_roofline(seq_len, batch, tp, ep, args, gpu)

        # Store results
        results['batch_sizes'].append(batch)
        results['total_time'].append(mla_result['total_time_with_tp'])
        results['gemm_time'].append(mla_result['gemm_fp8_time'])
        results['attn_time'].append(mla_result['attn_fp16_time'])
        results['load_weight_time'].append(mla_result['load_weight_time'])
        results['load_kv_time'].append(mla_result['load_kv_time'])
        results['all_reduce_time'].append(mla_result['all_reduce_time'])

        # Print results
        print(f"  Total time (with TP): {mla_result['total_time_with_tp']:.3f} ms")
        print(f"  GEMM time: {mla_result['gemm_fp8_time']:.3f} ms")
        print(f"  Attention time: {mla_result['attn_fp16_time']:.3f} ms")

    return results


def visualize_batch_sweep(results, gpu: GPU_perf, seq_len, tp, ep, save_path=None, show_plot=False):
    """
    Visualize batch sweep results showing attention total time, gemm time, and attention time.

    Args:
        results: Dictionary containing batch sizes and metrics
        gpu: GPU performance parameters
        seq_len: Sequence length used
        tp: Tensor parallelism degree
        ep: Expert parallelism degree
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    batch_sizes = results['batch_sizes']
    total_time = results['total_time']
    gemm_time = results['gemm_time']
    attn_time = results['attn_time']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Main metrics (Total, GEMM, Attention)
    ax1.plot(batch_sizes, total_time, 'o-', linewidth=2.5, markersize=8,
             label='Total Time (with TP)', color='blue')
    ax1.plot(batch_sizes, gemm_time, 's-', linewidth=2.5, markersize=8,
             label='GEMM Time (FP8)', color='green')
    ax1.plot(batch_sizes, attn_time, '^-', linewidth=2.5, markersize=8,
             label='Attention Time (FP16)', color='red')

    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title(f'MLA Decode Time vs Batch Size\n{gpu.gpu_type} | seq_len={seq_len}, TP={tp}, EP={ep}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_xticks(batch_sizes)

    # Subplot 2: Stacked bar chart showing time breakdown
    load_weight_time = results['load_weight_time']
    load_kv_time = results['load_kv_time']
    all_reduce_time = results['all_reduce_time']

    x_pos = np.arange(len(batch_sizes))
    width = 0.6

    # Create stacked bars
    p1 = ax2.bar(x_pos, gemm_time, width, label='GEMM Time', color='green', alpha=0.8)
    p2 = ax2.bar(x_pos, attn_time, width, bottom=gemm_time,
                 label='Attention Time', color='red', alpha=0.8)

    # Add AllReduce if TP > 1
    if tp > 1:
        bottom = [g + a for g, a in zip(gemm_time, attn_time)]
        p3 = ax2.bar(x_pos, all_reduce_time, width, bottom=bottom,
                     label='AllReduce Time', color='orange', alpha=0.8)

    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Time Breakdown by Component\n{gpu.gpu_type} | seq_len={seq_len}, TP={tp}, EP={ep}',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()

    plt.close()

    return fig


def visualize_batch_sweep_detailed(results, gpu: GPU_perf, seq_len, tp, ep, save_path=None, show_plot=False):
    """
    Create detailed visualization with 3 subplots showing all metrics.

    Args:
        results: Dictionary containing batch sizes and metrics
        gpu: GPU performance parameters
        seq_len: Sequence length used
        tp: Tensor parallelism degree
        ep: Expert parallelism degree
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    batch_sizes = results['batch_sizes']

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))

    # Subplot 1: All times
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(batch_sizes, results['total_time'], 'o-', linewidth=2.5, markersize=8,
             label='Total Time', color='blue')
    ax1.plot(batch_sizes, results['gemm_time'], 's-', linewidth=2.5, markersize=8,
             label='GEMM Time', color='green')
    ax1.plot(batch_sizes, results['attn_time'], '^-', linewidth=2.5, markersize=8,
             label='Attention Time', color='red')

    ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Main Metrics', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    ax1.set_xticks(batch_sizes)

    # Subplot 2: Memory loading times
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(batch_sizes, results['load_weight_time'], 'D-', linewidth=2.5, markersize=8,
             label='Load Weight Time', color='purple')
    ax2.plot(batch_sizes, results['load_kv_time'], 'v-', linewidth=2.5, markersize=8,
             label='Load KV Cache Time', color='orange')

    ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Memory Loading Times', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    ax2.set_xticks(batch_sizes)

    # Subplot 3: Communication time or percentage breakdown
    ax3 = plt.subplot(1, 3, 3)
    if tp > 1:
        ax3.plot(batch_sizes, results['all_reduce_time'], 'p-', linewidth=2.5, markersize=8,
                 label='AllReduce Time', color='brown')
        ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        ax3.set_title('Communication Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=9)
        ax3.set_xticks(batch_sizes)
    else:
        # If TP=1, show percentage breakdown instead
        total = np.array(results['total_time'])
        gemm_pct = np.array(results['gemm_time']) / total * 100
        attn_pct = np.array(results['attn_time']) / total * 100

        ax3.plot(batch_sizes, gemm_pct, 's-', linewidth=2.5, markersize=8,
                 label='GEMM %', color='green')
        ax3.plot(batch_sizes, attn_pct, '^-', linewidth=2.5, markersize=8,
                 label='Attention %', color='red')
        ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Time Percentage Breakdown', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=9)
        ax3.set_xticks(batch_sizes)

    plt.suptitle(f'Batch Size Sweep Analysis - {gpu.gpu_type}\nseq_len={seq_len}, TP={tp}, EP={ep}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDetailed visualization saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()

    plt.close()

    return fig


def generate_sweep_plots(args: ModelArgs, gpu_dict: dict):
    """
    Generate sweep plots for different TP/EP configurations.

    Args:
        args: Model arguments
        gpu_dict: Dictionary of GPU configurations
    """
    # Define sweep parameters
    seq_lens = [1024, 2048, 4096, 8192]
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]

    # Define TP/EP configurations to test
    configs = [
        {'tp': 1, 'ep': 32, 'name': 'tp1_ep32'},
        {'tp': 4, 'ep': 32, 'name': 'tp4_ep32'},
        {'tp': 8, 'ep': 32, 'name': 'tp8_ep32'},
        {'tp': 8, 'ep': 64, 'name': 'tp8_ep64'},
    ]

    # Test GPUs
    test_gpus = ['H800', 'DGX-B200']

    for gpu_name in test_gpus:
        if gpu_name not in gpu_dict:
            print(f"Warning: {gpu_name} not found, skipping...")
            continue

        gpu = gpu_dict[gpu_name]

        for config in configs:
            tp = config['tp']
            ep = config['ep']
            config_name = config['name']

            print(f"\n{'='*80}")
            print(f"Generating sweep plot: {gpu_name} - TP={tp}, EP={ep}")
            print(f"{'='*80}")

            save_filename = f"decode_roofline_sweep_{gpu_name.lower().replace('-', '_')}_{config_name}.png"

            visualize_decode_sweeps(
                seq_lens=seq_lens,
                batch_sizes=batch_sizes,
                tp=tp,
                ep=ep,
                args=args,
                gpu=gpu,
                title=f"Decode Stage Roofline Sweep - {gpu_name}",
                save_path=save_filename,
                show_plot=False
            )

            print(f"Saved: {save_filename}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create model args
    args = ModelArgs()

    # Load GPU configurations from CSV
    print("=" * 80)
    print("Loading GPU configurations from device/gpu_info.csv")
    print("=" * 80)

    gpu_dict = get_gpu_info(
        filename='./device/gpu_info.csv',
        discount_rate=0.85,
        decoding_mode=True,  # Set comm_sm=0 for decode mode
        print_console=True
    )

    # Select GPUs to test
    test_gpus = ['H800', 'H200', 'DGX-B200', 'GB200-NVL72']

    # Configuration
    seq_len = 9999999999
    batch = 64
    tp = 8
    ep = 32

    print("\n" + "=" * 80)
    print("Part 1: Single Configuration Analysis")
    print("=" * 80)

    # Test selected GPUs
    for gpu_name in test_gpus:
        if gpu_name not in gpu_dict:
            print(f"\nWarning: {gpu_name} not found in GPU database, skipping...")
            continue

        gpu = gpu_dict[gpu_name]

        print("\n" + "=" * 80)
        print(f"Decode Stage Roofline Analysis - {gpu_name}")
        print(f"seq_len={seq_len}, batch={batch}, tp={tp}, ep={ep}")
        print("=" * 80)

        # MLA analysis
        mla_result = decode_mla_roofline(seq_len, batch, tp, ep, args, gpu)
        print(f"\nMLA Results ({gpu_name}):")
        for key, value in mla_result.items():
            print(f"  {key}: {value:.3f} ms")

        # MOE analysis
        moe_result = decode_moe_roofline(seq_len, batch, tp, ep, args, gpu)
        print(f"\nMOE Results ({gpu_name}):")
        for key, value in moe_result.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

        # Visualize roofline
        print(f"\nGenerating Roofline Visualization for {gpu_name}...")

        save_filename = f"decode_roofline_{gpu_name.lower().replace('-', '_')}.png"
        visualize_decode_operations(
            seq_len=seq_len,
            batch=batch,
            tp=tp,
            ep=ep,
            args=args,
            gpu=gpu,
            title="Decode Stage Roofline Analysis",
            save_path=save_filename,
            show_plot=False  # Set to False for CLI execution
        )

        print(f"Visualization saved as '{save_filename}'\n")

    # Example 2: Compare different batch sizes on H800
    if 'H800' in gpu_dict:
        print("=" * 80)
        print("Comparing Different Batch Sizes on H800...")
        print("=" * 80)

        batch_sizes = [16, 32, 64, 128]

        for bs in batch_sizes:
            print(f"\n--- Batch Size: {bs} ---")
            mla = decode_mla_roofline(seq_len, bs, tp, ep, args, gpu_dict['H800'])
            moe = decode_moe_roofline(seq_len, bs, tp, ep, args, gpu_dict['H800'])
            print(f"MLA total time (TP={tp}): {mla['total_time_with_tp']:.3f} ms")
            print(f"MOE shared time: {moe['shared_expert_time']:.3f} ms")
            print(f"MOE routed time: {moe['routed_expert_time']:.3f} ms")

    # Part 2: Batch size sweep analysis with fixed parameters
    print("\n" + "=" * 80)
    print("Part 2: Batch Size Sweep Analysis (fixed seq_len, tp, ep)")
    print("=" * 80)

    # Configuration for batch sweep
    fixed_seq_len = 8192
    fixed_tp = 8
    fixed_ep = 32
    batch_sizes_to_test = [16, 32, 64, 128, 256, 512]

    # Test on H800
    if 'H800' in gpu_dict:
        print(f"\nRunning batch size sweep on H800...")
        results = batch_sweep_analysis(
            seq_len=fixed_seq_len,
            tp=fixed_tp,
            ep=fixed_ep,
            batch_sizes=batch_sizes_to_test,
            args=args,
            gpu=gpu_dict['H800']
        )

        # Generate visualizations
        save_path_main = f'batch_sweep_h800_seq{fixed_seq_len}_tp{fixed_tp}_ep{fixed_ep}.png'
        visualize_batch_sweep(results, gpu_dict['H800'], fixed_seq_len, fixed_tp, fixed_ep,
                            save_path=save_path_main, show_plot=False)

        save_path_detailed = f'batch_sweep_detailed_h800_seq{fixed_seq_len}_tp{fixed_tp}_ep{fixed_ep}.png'
        visualize_batch_sweep_detailed(results, gpu_dict['H800'], fixed_seq_len, fixed_tp, fixed_ep,
                                     save_path=save_path_detailed, show_plot=False)

        # Print summary table
        print("\n" + "=" * 80)
        print("Summary Table:")
        print("-" * 80)
        print(f"{'Batch':>8} | {'Total (ms)':>12} | {'GEMM (ms)':>12} | {'Attn (ms)':>12} | {'AllReduce (ms)':>15}")
        print("-" * 80)
        for i, batch in enumerate(results['batch_sizes']):
            print(f"{batch:>8} | {results['total_time'][i]:>12.3f} | "
                  f"{results['gemm_time'][i]:>12.3f} | {results['attn_time'][i]:>12.3f} | "
                  f"{results['all_reduce_time'][i]:>15.3f}")
        print("-" * 80)

    # Part 3: Generate sweep plots for different TP/EP configurations
    print("\n" + "=" * 80)
    print("Part 3: Sweep Analysis (varying seq_len and batch_size)")
    print("=" * 80)

    generate_sweep_plots(args, gpu_dict)

    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)

