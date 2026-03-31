#include "rabitq_distance.hpp"
#include <cmath>

namespace rabitq {

uint32_t ip_byte_bin(const uint64_t* code, const uint64_t* bit_planes, uint32_t words) {
    uint32_t result = 0;
    for (int j = 0; j < B_QUERY; j++) {
        uint32_t plane_ip = 0;
        for (uint32_t w = 0; w < words; w++) {
            plane_ip += __builtin_popcountll(code[w] & bit_planes[j * words + w]);
        }
        result += plane_ip << j;  // Weight by 2^j
    }
    return result;
}

DistanceEstimate estimate_distance(
    const QuantizedVector& qv,
    const QuantizedQuery& qq,
    uint32_t B)
{
    DistanceEstimate est;
    const uint32_t words = words_per_code(B);
    const float sqrt_B = std::sqrt(static_cast<float>(B));

    // Compute ⟨x̄_b, q̄_u⟩ via bitwise operations
    uint32_t raw_ip = ip_byte_bin(qv.code.data(), qq.bit_planes.data(), words);

    // Reconstruct ⟨x̄, q̄⟩ from the efficient formula (matching reference):
    // ⟨x̄, q̄⟩ = (2Δ/√B)·⟨x̄_b, q̄_u⟩ + (2v_l/√B)·Σx̄_b[i] - (Δ/√B)·Σq̄_u[i] - √B·v_l
    float x_bar_q_bar =
        (2.0f * qq.width / sqrt_B) * static_cast<float>(raw_ip)
        + (2.0f * qq.vl / sqrt_B) * static_cast<float>(qv.popcount)
        - (qq.width / sqrt_B) * static_cast<float>(qq.sum_q)
        - sqrt_B * qq.vl;

    // Unbiased estimator: ⟨x̄, q̄⟩/x0 ≈ ||q_r-c|| · ⟨o, q⟩
    // (q̄ approximates q' = P^{-1}(q_r-c) which is NOT unit-normalized)
    float scaled_ip = x_bar_q_bar / qv.x0;
    float dist_o = qv.dist_to_centroid;
    float dist_q = std::sqrt(qq.sqr_y);

    // Unit inner product estimate (for reporting)
    est.estimated_ip = (dist_q > 1e-10f) ? (scaled_ip / dist_q) : 0.0f;

    // Squared distance (matching reference: sqr_x + sqr_y - 2·dist_o·⟨x̄,q̄⟩/x0)
    est.estimated_dist = dist_o * dist_o + qq.sqr_y - 2.0f * dist_o * scaled_ip;

    // Error bound: 2·dist_q·dist_o·√((1-x0²)/x0²)·ε₀/√(B-1)
    float x0_sq = qv.x0 * qv.x0;
    float ip_error = std::sqrt((1.0f - x0_sq) / x0_sq) * EPSILON_0 / std::sqrt(static_cast<float>(B - 1));
    est.error_bound = 2.0f * dist_o * dist_q * ip_error;

    // Lower bound on distance (optimistic: subtract error)
    est.lower_bound_dist = est.estimated_dist - est.error_bound;

    return est;
}

}  // namespace rabitq
