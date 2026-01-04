# FlashAttention v1 (emulated) in pure PyTorch: blockwise attention with online softmax.
# This demonstrates the FA-v1 algorithmic idea without requiring custom CUDA kernels.
# It compares against a naive attention implementation for correctness and timing.

import math
import time
import torch

def naive_attention(q, k, v, causal: bool):
    """
    q, k, v: (B, H, N, D)
    returns: (B, H, N, D)
    """
    import math, torch
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # 用 float32 做 scores/softmax 更稳健
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale  # (B,H,N,N) float32

    if causal:
        i = torch.arange(N, device=scores.device)
        # mask=True 的位置要被屏蔽（k_pos > q_pos）
        mask = i[None, None, :, None] < i[None, None, None, :]
        scores = scores.masked_fill(mask, float('-inf'))  # 用 -inf，避免半精度溢出

    p = torch.softmax(scores, dim=-1)                     # float32
    out = torch.matmul(p, v.float())                      # float32
    return out.to(q.dtype)                                # 回到原 dtype（如 float16）



@torch.no_grad()
def flashattention_v1_emulated(q, k, v, causal: bool, q_block: int = 128, k_block: int = 256):
    """
    Emulates FlashAttention v1 using blockwise IO-aware computation and online softmax.
    q, k, v: (B, H, N, D)
    returns: (B, H, N, D)
    """
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    device = q.device
    out = torch.empty_like(q)

    # Process Q in blocks to control working set size
    for qs in range(0, N, q_block):
        qe = min(qs + q_block, N)
        q_blk = q[:, :, qs:qe, :]  # (B, H, qB, D)
        qB = q_blk.shape[2]

        # Online softmax running stats for each row in this Q block
        # m: running max; l: running sum of exp; o: running output numerator
        # Use large negative sentinel instead of -inf for numeric stability in exp() arithmetic
        # 1) 初始化 m, l, o
        m = torch.full((B, H, qB), float('-inf'), device=device, dtype=torch.float32)  # 用 -inf
        l = torch.zeros((B, H, qB), device=device, dtype=torch.float32)
        o = torch.zeros((B, H, qB, D), device=device, dtype=torch.float32)

        # Iterate over K/V blocks
        for ks in range(0, N, k_block):
            ke = min(ks + k_block, N)
            k_blk = k[:, :, ks:ke, :]  # (B, H, kB, D)
            v_blk = v[:, :, ks:ke, :]  # (B, H, kB, D)
            kB = k_blk.shape[2]

            # scores: (B, H, qB, kB)
            scores = torch.matmul(q_blk.float(), k_blk.float().transpose(-1, -2)) * scale

            # 2) 分块内的因果掩码
            if causal:
                q_pos = torch.arange(qs, qe, device=device)
                k_pos = torch.arange(ks, ke, device=device)
                valid = (q_pos[:, None] >= k_pos[None, :])  # (qB,kB)
                # scores 是 float32，这里用 -inf
                scores = torch.where(valid[None, None, :, :], scores, torch.full_like(scores, float('-inf')))

            # Online softmax update
            s_max = scores.max(dim=-1).values  # (B, H, qB)
            m_new = torch.maximum(m, s_max)    # (B, H, qB)

            # scale old accumulators into the new max domain
            scale_old = torch.exp(m - m_new)   # (B, H, qB)
            # compute new partial probabilities under m_new
            p = torch.exp(scores - m_new[..., None])  # (B, H, qB, kB)

            # update l and o
            l = l * scale_old + p.sum(dim=-1)                     # (B, H, qB)
            o = o * scale_old[..., None] + torch.matmul(p, v_blk) # (B, H, qB, D)

            # commit new max
            m = m_new

        # finalize: normalize
        out[:, :, qs:qe, :] = (o / l[..., None]).to(q.dtype)

    return out


def run_demo(B=1, H=4, N=1024, D=64, causal=True, q_block=128, k_block=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Config -> B={B}, H={H}, N={N}, D={D}, causal={causal}, q_block={q_block}, k_block={k_block}")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, H, N, D, generator=g, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, generator=g, device=device, dtype=dtype)

    # Baseline timing
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ref = naive_attention(q, k, v, causal=causal)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Emulated FA-v1 timing
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    out_fa = flashattention_v1_emulated(q, k, v, causal=causal, q_block=q_block, k_block=k_block)
    if device == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    # Correctness check
    diff = (out_ref - out_fa).float()
    max_abs_err = diff.abs().max().item()
    rel_l2 = (diff.pow(2).sum() / (out_ref.float().pow(2).sum() + 1e-12)).sqrt().item()

    print(f"\nCorrectness: max_abs_err={max_abs_err:.3e}, rel_l2={rel_l2:.3e}")
    print(f"Timing (seconds): naive={t1 - t0:.4f}, FA-v1-emulated={t3 - t2:.4f} (smaller is better)")

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak CUDA memory (approx): {peak_mem:.1f} MB")

    return {
        "max_abs_err": max_abs_err,
        "rel_l2": rel_l2,
        "t_naive": t1 - t0,
        "t_fa": t3 - t2,
    }


# Run a few demos with modest sizes to keep it fast on CPU while still illustrative.
res1 = run_demo(N=512, D=64, H=4, q_block=128, k_block=256)    # small
res2 = run_demo(N=1024, D=64, H=4, q_block=128, k_block=256)   # medium
print("\nDone.")
