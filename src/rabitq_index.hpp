#pragma once

#include "defines.hpp"
#include "rotation.hpp"
#include <vector>

namespace rabitq {

/// Per-vector quantization result from the index phase.
struct QuantizedVector {
    std::vector<uint64_t> code;  // D-bit binary code packed into uint64_t words
    float dist_to_centroid;      // ||o_r - c||
    float x0;                    // ⟨ō, o⟩ (inner product between quantized and original)
    uint32_t popcount;           // Number of 1-bits in code (= Σ x̄_b[i])
};

/// Result of the index phase for a cluster of vectors.
struct IndexResult {
    Eigen::MatrixXf P;                      // Orthogonal rotation matrix [B, B]
    Eigen::VectorXf centroid;               // Cluster centroid [D]
    std::vector<QuantizedVector> vectors;   // Per-vector quantization data
    uint32_t D;                             // Original dimension
    uint32_t B;                             // Padded dimension (multiple of 64)
};

/// Compute the centroid (mean) of a set of row vectors.
///
/// @param X  Data matrix [N, D], each row is a vector
/// @return   Centroid vector [D]
Eigen::VectorXf compute_centroid(const Eigen::MatrixXf& X);

/// Run the RaBitQ index phase on a cluster of vectors.
///
/// Steps (matching rabitq.py reference):
///   1. Compute centroid c = mean(X)
///   2. Generate random orthogonal matrix P of size [B, B]
///   3. For each vector o_r:
///      a. Center: o_r - c
///      b. Inverse-rotate: v = (o_r - c) @ P^T  (in padded space)
///      c. Extract signs: x̄_b[i] = (v[i] > 0)
///      d. Compute dist_to_centroid = ||o_r - c||
///      e. Compute x0 = ⟨ō, o⟩ = ||v||_1 / √B (normalized)
///      f. Pack bits into uint64_t words
///
/// @param X     Data matrix [N, D], each row is a vector
/// @param seed  Random seed for rotation matrix (0 = random)
/// @return      IndexResult with P, centroid, and per-vector data
IndexResult build_index(const Eigen::MatrixXf& X, uint64_t seed = 0);

}  // namespace rabitq
