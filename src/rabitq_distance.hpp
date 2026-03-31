#pragma once

#include "defines.hpp"
#include "rabitq_index.hpp"
#include "rabitq_query.hpp"

namespace rabitq {

/// Result of a single distance estimation between a data vector and a query.
struct DistanceEstimate {
    float estimated_ip;     // Estimated inner product ⟨o, q⟩ (unbiased)
    float estimated_dist;   // Estimated squared distance ||o_r - q_r||²
    float lower_bound_dist; // Lower bound on squared distance (for re-ranking)
    float error_bound;      // Error bound on the IP estimate
};

/// Compute the binary-integer inner product ⟨x̄_b, q̄_u⟩ via bit planes.
///
/// Uses the decomposition: ⟨x̄_b, q̄_u⟩ = Σ_{j=0}^{B_q-1} 2^j · popcount(x̄_b & q̄_u^(j))
///
/// @param code       Binary code of the data vector [words_per_code(B) uint64_t]
/// @param bit_planes Query bit planes [B_QUERY * words_per_code(B) uint64_t]
/// @param words      Number of uint64_t words per code
/// @return           ⟨x̄_b, q̄_u⟩ (unsigned integer)
uint32_t ip_byte_bin(const uint64_t* code, const uint64_t* bit_planes, uint32_t words);

/// Estimate the squared distance between a data vector and a query.
///
/// Implements the full RaBitQ estimator (matching ivf_rabitq.h reference):
///   est_ip = ⟨x̄, q̄⟩ / ⟨ō, o⟩  (unbiased inner product estimate)
///   est_dist = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·est_ip
///   error_bound uses ε₀ = 1.9 and per-vector ⟨ō,o⟩
///
/// @param qv    Quantized data vector (from build_index)
/// @param qq    Quantized query (from quantize_query)
/// @param B     Padded dimension
/// @return      DistanceEstimate with estimated distance and error bound
DistanceEstimate estimate_distance(
    const QuantizedVector& qv,
    const QuantizedQuery& qq,
    uint32_t B);

}  // namespace rabitq
