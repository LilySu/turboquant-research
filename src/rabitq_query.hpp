#pragma once

#include "defines.hpp"
#include "rabitq_index.hpp"
#include <vector>

namespace rabitq {

/// Result of quantizing a query vector for a specific cluster.
struct QuantizedQuery {
    std::vector<uint8_t> q_u;       // B_q-bit quantized unsigned integers [B]
    std::vector<uint64_t> bit_planes; // B_q bit planes, each B bits [B_QUERY * words_per_code(B)]
    float vl;                        // Lower bound of quantization range
    float vr;                        // Upper bound of quantization range
    float width;                     // Step size Δ = (vr - vl) / (2^B_q - 1)
    uint32_t sum_q;                  // Σ q̄_u[i]
    float sqr_y;                     // ||q_r - c||² (squared distance to centroid)
};

/// Process a query vector for distance estimation against a cluster.
///
/// Steps (matching space.h reference):
///   1. Compute q' = P^{-1}(q_r - c) via inverse rotation
///   2. Find range: vl = min(q'), vr = max(q')
///   3. Uniform quantize each coordinate to B_q-bit unsigned integer:
///      q̄_u[i] = floor((q'[i] - vl) / Δ + u[i])  where u[i] ~ Uniform(0,1)
///   4. Extract B_q bit planes for bitwise distance computation
///   5. Precompute sum_q = Σ q̄_u[i]
///
/// @param query    Raw query vector [D]
/// @param index    IndexResult from build_index (provides P, centroid, B)
/// @param u        Random dither vector [B], each in [0,1). Pass empty for u=0.5 (deterministic).
/// @return         QuantizedQuery ready for distance computation
QuantizedQuery quantize_query(
    const Eigen::VectorXf& query,
    const IndexResult& index,
    const std::vector<float>& u = {});

/// Generate random dither vector for query quantization.
///
/// @param B     Padded dimension
/// @param seed  Random seed (0 = random device)
/// @return      Vector of B floats, each in [0, 1)
std::vector<float> generate_dither(uint32_t B, uint64_t seed = 0);

}  // namespace rabitq
