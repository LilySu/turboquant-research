# GLM5 KV Cache Architecture Analysis

## Overview

GLM5 (GLM-MoE-DSA) is a Mixture-of-Experts transformer with **Multi-head Latent Attention (MLA)** and **Dynamic Sparse Attention (DSA)**. This analysis is based on the local repository at `/home/lily/wsl_git/glm5`, specifically the `glm5-raw-decoupled-from-hf/` pure PyTorch implementation.

## Model Configuration

From `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/config.py`:

| Parameter | Value | Notes |
|:--|:--|:--|
| `hidden_size` | 6144 | Model dimension |
| `num_hidden_layers` | 78 | 3 dense + 75 sparse (MoE) |
| `num_attention_heads` | 64 | Number of query heads |
| `num_key_value_heads` | 64 | GQA ratio = 1 (no grouping) |
| `q_lora_rank` | 2048 | Query compression rank |
| `kv_lora_rank` | 512 | **KV compression rank** |
| `qk_rope_head_dim` | 64 | RoPE dimension per head |
| `qk_nope_head_dim` | 192 | Non-RoPE QK dimension |
| `qk_head_dim` | 256 | Total QK dim (192 + 64) |
| `v_head_dim` | 256 | Value head dimension |
| `index_topk` | 2048 | DSA sparse attention top-k |
| `max_position_embeddings` | 202752 | ~200K context window |

## MLA: Multi-head Latent Attention

MLA compresses the KV cache by projecting key-value pairs through a low-rank bottleneck **before** caching. This is fundamentally different from standard MHA where full K and V tensors are cached.

### Projection Chain

From `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/model.py:243-282`:

```python
# Step 1: Compress hidden_states → low-rank KV + RoPE keys
self.kv_a_proj_with_mqa = nn.Linear(
    hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
)  # 6144 → 576 (512 + 64)

# Step 2: Normalize compressed representation
self.kv_a_layernorm = RMSNorm(kv_lora_rank)  # Over 512-dim

# Step 3: Expand back to full multi-head K and V
self.kv_b_proj = nn.Linear(
    kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
)  # 512 → 64 × (192 + 256) = 28672
```

### Data Flow

```
hidden_states [B, S, 6144]
    │
    ├─ kv_a_proj_with_mqa ──→ [B, S, 576]
    │                          ├─ k_compressed [B, S, 512]  ← This is cached
    │                          └─ k_pe [B, S, 64]           ← RoPE stream, also cached
    │
    ├─ kv_a_layernorm(k_compressed) ──→ [B, S, 512]
    │
    └─ kv_b_proj ──→ [B, S, H*(192+256)] = [B, S, 28672]
                     ├─ k_nope [B, H, S, 192]
                     └─ value   [B, H, S, 256]
```

### What Gets Cached

**Standard MHA** would cache: K `[B, 64, S, 256]` + V `[B, 64, S, 256]` = **32,768** floats per token.

**MLA caches the compressed representation instead:**

| Cached Tensor | Shape | Size per Token | Notes |
|:--|:--|:--|:--|
| `k_compressed` | `[B, H, S, 512/H]` → via expand | See below | Low-rank bottleneck |
| `k_pe` | `[B, H, S, 64]` | 64 × H floats | RoPE stream |

In the raw implementation (model.py:296-297):
```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx
    )
```

The code caches the **expanded** K and V (after `kv_b_proj`), not the compressed form. This means:
- K cached: `[B, 64, S, 256]` — full key states
- V cached: `[B, 64, S, 256]` — full value states

**This is a naive implementation.** The FlashMLA variant (`glm5-kernels-flashmla-deepgemm/`) caches the compressed 576D representation directly.

## KV Cache Implementation

From `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/cache.py`:

```python
class KVCache:
    """Stores key/value tensors in [B, H, T, D] format per layer."""

    def __init__(self, num_layers: int):
        self._cache = [None] * num_layers  # List of (K, V) tuples

    def update(self, key_states, value_states, layer_idx):
        if self._cache[layer_idx] is not None:
            prev_k, prev_v = self._cache[layer_idx]
            key_states = torch.cat([prev_k, key_states], dim=2)  # Concat along seq
            value_states = torch.cat([prev_v, value_states], dim=2)
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states
```

| Aspect | Detail |
|:--|:--|
| **Format** | `[B, H, T, D]` (batch, heads, seq_len, head_dim) |
| **Growth** | Dynamic concatenation along dim=2 |
| **Eviction** | None — grows unbounded |
| **Dtype** | Same as input (typically BF16 or FP16) |
| **Per layer** | Independent K, V pair per layer |

### Memory Per Token (Raw Implementation)

| Component | Shape | Bytes (BF16) |
|:--|:--|:--|
| K per layer | `[1, 64, 1, 256]` | $64 \times 256 \times 2 = 32{,}768$ |
| V per layer | `[1, 64, 1, 256]` | $64 \times 256 \times 2 = 32{,}768$ |
| **KV per layer** | | **65,536** (64 KB) |
| **78 layers** | | **5,111,808** (~4.9 MB per token) |

At 200K tokens: **~960 GB** — far exceeding GPU memory. This is why MLA compression and DSA sparse attention are essential.

### FlashMLA Optimized Cache

From `/home/lily/wsl_git/glm5/glm5-kernels-flashmla-deepgemm/mla_attention.py:12-14`:

```
# KV cache = 576D (512 compressed nope + 64 BF16 rope)
# Weight absorption: kv_b_proj absorbed into Q and O projections
```

With weight absorption, the cache stores only the compressed 576D representation:

| Component | Shape | Bytes (BF16) |
|:--|:--|:--|
| Compressed KV per layer | `[1, 1, 1, 576]` | $576 \times 2 = 1{,}152$ |
| **78 layers** | | **89,856** (~88 KB per token) |

**Compression ratio:** 4.9 MB → 88 KB = **~56× reduction** from MLA alone.

## DSA: Dynamic Sparse Attention

From `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/model.py:139-209`:

DSA selects only the top-2048 most relevant tokens for each query position, reducing attention from $O(S^2)$ to $O(S \cdot 2048)$.

```python
class DSAIndexer(nn.Module):
    # Lightweight scorer: separate from main attention
    # index_n_heads=32 (fewer than attention heads)
    # index_head_dim=128 (smaller than qk_head_dim=256)
    # index_topk=2048

    def forward(self, hidden_states, q_resid, position_embeddings, ...):
        # Score all tokens, select top-2048
        topk_indices = index_scores.topk(topk, dim=-1).indices  # [B, S, 2048]
```

The indexer maintains its **own key cache** (`_cached_keys`), separate from the main KV cache.

## Existing Quantization/Compression

| File | Quantization Type |
|:--|:--|
| `glm5-kernels-tensorRT-llm-deepgemm-dsa/fp8_utils.py` | FP8 weight quantization for DeepGemm |
| `glm5-kernels-flashmla-deepgemm/mla_attention.py` | FP8 paged KV cache (FlashMLA) |
| `benchmark/fp8_pareto/bench_fp8.py` | FP8 benchmarking |

The FlashMLA variant supports **FP8 KV cache** natively — the paged cache stores compressed KV in FP8 format. No sub-8-bit KV cache quantization exists in the codebase.

## Hook Points for TurboQuant/RaBitQ Integration

### Option A: Quantize the Compressed KV (Recommended)

Target the 576D compressed representation before caching:

```python
# In MLAttention.forward(), after line 277:
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, 576]

# INSERT HERE: TurboQuant quantization of compressed_kv
# compressed_kv_quant = turboquant_encode(compressed_kv)  # [B, S, 576] → quantized

# On read: dequantize before kv_b_proj expansion
# compressed_kv = turboquant_decode(compressed_kv_quant)
```

**File:** `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/model.py:276`

**Advantages:** Quantizes 576D vectors instead of 256D per-head vectors; MLA's compression already removes redundancy, so TurboQuant operates on a clean signal.

### Option B: Quantize the Expanded K/V in the Cache

Target the `KVCache.update()` method:

```python
# In cache.py, modify update():
def update(self, key_states, value_states, layer_idx):
    # key_states: [B, H, S, 256], value_states: [B, H, S, 256]
    # INSERT: quantize before storing
    # key_states_q = turboquant_encode(key_states)
    # value_states_q = turboquant_encode(value_states)
    ...
```

**File:** `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/cache.py:16`

**Advantages:** Simpler integration; works with any attention variant.

### Option C: Replace FP8 with TurboQuant in FlashMLA Cache

Target the paged KV cache in the FlashMLA variant:

**File:** `/home/lily/wsl_git/glm5/glm5-kernels-flashmla-deepgemm/cache.py`

This requires deeper integration with the CUDA kernel but could provide the best performance.

## Key Takeaways

- GLM5 uses **MLA** to compress KV from 32K → 576 floats per token per layer (56× reduction)
- The raw implementation caches expanded K/V; the FlashMLA variant caches compressed 576D
- **DSA** further reduces attention cost by selecting top-2048 tokens per query
- Existing FP8 quantization covers weights and FlashMLA KV cache
- **Best TurboQuant hook point:** After `kv_a_proj_with_mqa` (line 276 in model.py) — quantize the 576D compressed representation
- The 576D compressed vector has $D = 576$; TurboQuant with $b = 2$ bits would reduce KV memory by another ~8× beyond MLA's compression

## Sources

- `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/config.py` — Model configuration
- `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/model.py` — MLAttention, DSAIndexer, DecoderLayer
- `/home/lily/wsl_git/glm5/glm5-raw-decoupled-from-hf/cache.py` — KVCache implementation
- `/home/lily/wsl_git/glm5/glm5-kernels-flashmla-deepgemm/mla_attention.py` — FlashMLA + FP8 KV cache
- `/home/lily/wsl_git/glm5/glm5-kernels-tensorRT-llm-deepgemm-dsa/fp8_utils.py` — FP8 quantization
