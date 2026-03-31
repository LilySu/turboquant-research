#include "rabitq_query.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace rabitq {

std::vector<float> generate_dither(uint32_t B, uint64_t seed) {
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> u(B);
    for (uint32_t i = 0; i < B; i++)
        u[i] = dist(rng);
    return u;
}

QuantizedQuery quantize_query(
    const Eigen::VectorXf& query,
    const IndexResult& index,
    const std::vector<float>& u)
{
    const uint32_t D = index.D;
    const uint32_t B = index.B;
    const uint32_t max_val = (1 << B_QUERY) - 1;  // 15 for B_q=4

    QuantizedQuery result;

    // Step 1: Pad query and centroid, compute centered rotated query
    Eigen::VectorXf q_pad = Eigen::VectorXf::Zero(B);
    q_pad.head(D) = query;

    Eigen::VectorXf c_pad = Eigen::VectorXf::Zero(B);
    c_pad.head(D) = index.centroid;

    // q' = (q_pad - c_pad) @ P^T  (inverse rotation)
    Eigen::VectorXf centered = q_pad - c_pad;
    Eigen::RowVectorXf q_prime = centered.transpose() * index.P;  // row @ P = P^T @ col in effect

    // Compute sqr_y = ||q_r - c||²
    result.sqr_y = centered.head(D).squaredNorm();

    // Step 2: Compute range
    result.vl = q_prime.minCoeff();
    result.vr = q_prime.maxCoeff();
    result.width = (result.vr - result.vl) / static_cast<float>(max_val);

    // Step 3: Uniform scalar quantization with optional randomized rounding
    float one_over_width = (result.width > 1e-10f) ? (1.0f / result.width) : 0.0f;
    result.q_u.resize(B);
    result.sum_q = 0;

    bool use_dither = !u.empty();
    for (uint32_t i = 0; i < B; i++) {
        float val = (q_prime(i) - result.vl) * one_over_width;
        if (use_dither) {
            val += u[i];
        } else {
            val += 0.5f;  // Deterministic midpoint rounding
        }
        uint32_t q_val = static_cast<uint32_t>(val);
        if (q_val > max_val) q_val = max_val;
        result.q_u[i] = static_cast<uint8_t>(q_val);
        result.sum_q += q_val;
    }

    // Step 4: Extract B_QUERY bit planes
    // Bit plane j contains the j-th bit of each q_u[i], packed into uint64_t words
    const uint32_t words = words_per_code(B);
    result.bit_planes.resize(B_QUERY * words, 0);

    for (uint32_t i = 0; i < B; i++) {
        uint8_t val = result.q_u[i];
        for (int j = 0; j < B_QUERY; j++) {
            if ((val >> j) & 1) {
                result.bit_planes[j * words + i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    return result;
}

}  // namespace rabitq
