# DeepSeekV3.2 KV Cache Architecture Analysis

## Overview

DeepSeekV3.2 is a 671B-parameter Mixture-of-Experts model (~37B active per token) using **Multi-head Latent Attention (MLA)** — the same attention mechanism as GLM5 but **without** Dynamic Sparse Attention (DSA). This analysis is based on the local repository at `/home/lily/wsl_git/deepseekv3_2`, specifically the `deepseekv3_2-raw-decoupled-from-hf/` pure PyTorch implementation.

## Model Configuration

From `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/config.py`:

| Parameter | DeepSeekV3.2 | GLM5 | Notes |
|:--|:--|:--|:--|
| `hidden_size` | 7168 | 6144 | DSV3 is wider |
| `num_hidden_layers` | 61 | 78 | DSV3 has fewer layers |
| `num_attention_heads` | 128 | 64 | DSV3 has 2× more heads |
| `num_key_value_heads` | 128 | 64 | GQA ratio = 1 (both) |
| `q_lora_rank` | 1536 | 2048 | Query compression |
| `kv_lora_rank` | **512** | **512** | Same KV bottleneck |
| `qk_rope_head_dim` | 64 | 64 | Same RoPE dim |
| `qk_nope_head_dim` | 128 | 192 | DSV3 smaller nope |
| `qk_head_dim` | 192 | 256 | DSV3 smaller head |
| `v_head_dim` | 128 | 256 | DSV3 half the V dim |
| `max_position_embeddings` | 163,840 | 202,752 | Both ~160-200K |
| DSA | **No** | Yes (top-2048) | Key architectural difference |

## MLA Projection Chain

From `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/model.py:237-244`:

```python
# Step 1: Compress to low-rank KV + RoPE keys
self.kv_a_proj_with_mqa = nn.Linear(
    hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
)  # 7168 → 576 (512 + 64)

# Step 2: Normalize
self.kv_a_layernorm = RMSNorm(kv_lora_rank)  # Over 512-dim

# Step 3: Expand to full multi-head K and V
self.kv_b_proj = nn.Linear(
    kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
)  # 512 → 128 × (128 + 128) = 32768
```

### Data Flow

```
hidden_states [B, S, 7168]
    │
    ├─ kv_a_proj_with_mqa ──→ [B, S, 576]
    │                          ├─ k_compressed [B, S, 512]  ← Cached (compressed)
    │                          └─ k_pe [B, S, 64]           ← RoPE stream
    │
    ├─ kv_a_layernorm(k_compressed) ──→ [B, S, 512]
    │
    └─ kv_b_proj ──→ [B, S, H*(128+128)] = [B, S, 32768]
                     ├─ k_nope [B, 128, S, 128]
                     └─ value  [B, 128, S, 128]
```

**Identical MLA bottleneck as GLM5:** Both compress to 576D (512 + 64) before caching. The difference is in the expansion — DeepSeekV3.2 uses 128 heads with smaller head dimensions (128 vs 192/256).

## KV Cache Implementation

From `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/cache.py` — **identical** to GLM5:

```python
class KVCache:
    """Stores key/value tensors in [B, H, T, D] format per layer."""

    def update(self, key_states, value_states, layer_idx):
        if self._cache[layer_idx] is not None:
            prev_k, prev_v = self._cache[layer_idx]
            key_states = torch.cat([prev_k, key_states], dim=2)
            value_states = torch.cat([prev_v, value_states], dim=2)
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states
```

### What Gets Cached (Raw Implementation)

The raw code (model.py:285-286) caches **expanded** K and V:

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx  # [B, 128, S, 192] and [B, 128, S, 128]
    )
```

### Memory Per Token

**Raw (expanded K/V):**

| Component | Shape | Bytes (BF16) |
|:--|:--|:--|
| K per layer | `[1, 128, 1, 192]` | $128 \times 192 \times 2 = 49{,}152$ |
| V per layer | `[1, 128, 1, 128]` | $128 \times 128 \times 2 = 32{,}768$ |
| **KV per layer** | | **81,920** (80 KB) |
| **61 layers** | | **4,997,120** (~4.8 MB per token) |

**MLA compressed (with weight absorption):**

| Component | Shape | Bytes (BF16) |
|:--|:--|:--|
| Compressed KV per layer | `[1, 1, 1, 576]` | $576 \times 2 = 1{,}152$ |
| **61 layers** | | **70,272** (~69 KB per token) |

**Compression ratio:** 4.8 MB → 69 KB = **~70× reduction** from MLA alone.

## How MLA Differs from Standard MHA

| Aspect | Standard MHA | MLA (DeepSeekV3.2 / GLM5) |
|:--|:--|:--|
| **Cache contents** | Full K `[B,H,S,d_k]`, V `[B,H,S,d_v]` | Compressed `[B,1,S,576]` |
| **Cache size/token** | $2 \cdot H \cdot d \cdot 2$ bytes | $576 \cdot 2$ bytes |
| **Cache overhead (61L)** | ~4.8 MB | ~69 KB |
| **KV projection** | Direct `W_K`, `W_V` from hidden | Two-stage: compress then expand |
| **Bottleneck** | None | `kv_lora_rank = 512` |
| **RoPE handling** | Applied to full K | Applied to separate 64D stream |
| **Weight absorption** | N/A | `kv_b_proj` absorbed into Q/O projections |

**Key insight for TurboQuant:** MLA's 576D compressed representation is the natural quantization target. This is a **single vector per token** (not per head), making it ideal for TurboQuant's rotation-based approach.

## No DSA in DeepSeekV3.2

Unlike GLM5, DeepSeekV3.2 uses **standard causal attention** (model.py:288-293):

```python
# Standard causal attention (no DSA sparse masking)
attn_output, attn_weights = eager_attention_forward(
    self, query_states, key_states, value_states, attention_mask, ...
)
```

This means DeepSeekV3.2 attends to **all** past tokens, making KV cache compression even more critical for long sequences. At 160K tokens without MLA compression, the cache would be ~768 GB.

## Existing Compression Mechanisms

| Mechanism | Where | Description |
|:--|:--|:--|
| **MLA compression** | `kv_a_proj_with_mqa` | Reduces 7168D → 576D before caching |
| **FlashMLA FP8 cache** | `deepseekv3_2-kernels-flashmla-deepgemm/` | FP8 paged KV cache |
| **Weight absorption** | FlashMLA variant | Absorbs `kv_b_proj` into Q/O, caching only 576D |
| **YaRN RoPE** | Config `rope_scaling` | Extends context via modified RoPE (not cache compression) |

No sub-8-bit KV cache quantization exists in the codebase.

## Hook Points for TurboQuant/RaBitQ Integration

### Option A: Quantize Compressed KV (Recommended)

Target the 576D compressed representation:

```python
# In MLAttention.forward(), after line 266:
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, 576]

# INSERT: TurboQuant quantization
# compressed_kv_quant = turboquant_encode(compressed_kv, b=2)  # 576 × 2 bits = 144 bytes
```

**File:** `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/model.py:266`

**Effect:** With $b = 2$ bits per coordinate: $576 \times 2 / 8 = 144$ bytes per token per layer. Across 61 layers: **8,784 bytes** (~8.6 KB per token) — a further **8× reduction** beyond MLA's compression.

### Option B: Quantize in KVCache.update()

Target the cache storage layer:

**File:** `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/cache.py:16`

Same approach as GLM5 — quantize K/V before storing, dequantize on read.

### Option C: Replace FP8 with TurboQuant in FlashMLA

Target the paged cache in the FlashMLA variant:

**File:** `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-kernels-flashmla-deepgemm/`

## Comparison: DeepSeekV3.2 vs GLM5 for TurboQuant

| Aspect | DeepSeekV3.2 | GLM5 |
|:--|:--|:--|
| **Compressed dim** | 576 (identical) | 576 (identical) |
| **Layers** | 61 | 78 |
| **Attention type** | Full causal | DSA (top-2048 sparse) |
| **TurboQuant benefit** | Higher (full attention needs more cache) | Still high, but DSA reduces attention scope |
| **Integration complexity** | Simpler (no DSA indexer cache) | DSA indexer has separate cache to consider |
| **FlashMLA variant** | Available with FP8 | Available with FP8 |

Both models share the **same MLA bottleneck dimension** (576D), so a single TurboQuant integration can target both with minimal adaptation.

## Key Takeaways

- DeepSeekV3.2 uses **MLA** (same as GLM5) with `kv_lora_rank = 512`, compressing KV to 576D (70× vs raw)
- **No DSA** — full causal attention makes KV cache compression even more critical
- Raw implementation caches expanded K/V `[B, 128, S, 192]`; FlashMLA caches compressed 576D
- **Best TurboQuant hook:** After `kv_a_proj_with_mqa` (model.py:266) — quantize 576D vectors
- With $b = 2$ TurboQuant: ~8.6 KB/token (vs 69 KB compressed, vs 4.8 MB raw)
- Same 576D bottleneck as GLM5 enables **shared TurboQuant integration** for both models

## Sources

- `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/config.py` — Model configuration
- `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/model.py` — MLAttention (lines 206-297)
- `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-raw-decoupled-from-hf/cache.py` — KVCache
- `/home/lily/wsl_git/deepseekv3_2/deepseekv3_2-kernels-flashmla-deepgemm/mla_attention.py` — FlashMLA variant
- [DeepSeek-V3 Technical Report — arXiv 2412.19437](https://arxiv.org/abs/2412.19437) — Architecture reference
