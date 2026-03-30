#pragma once

#include <Eigen/Dense>
#include <cstdint>

// Common type aliases for the RaBitQ implementation.
// Matches the reference: https://github.com/gaoj0017/RaBitQ

namespace rabitq {

// Matrix types (row-major for cache-friendly row iteration)
using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXf = Eigen::VectorXf;
using VectorXi = Eigen::VectorXi;

// Fixed-size types for common dimensions
template <int D>
using VectorDf = Eigen::Matrix<float, D, 1>;

template <int D>
using MatrixDf = Eigen::Matrix<float, Eigen::Dynamic, D, Eigen::RowMajor>;

// Binary code storage: D bits packed into uint64_t words
// For D=128: 2 uint64_t per vector. For D=960: 15 uint64_t per vector.
inline constexpr uint32_t words_per_code(uint32_t D) {
    return (D + 63) / 64;
}

// Padded dimension (round up to multiple of 64 for bit packing)
inline constexpr uint32_t pad_dim(uint32_t D) {
    return (D + 63) / 64 * 64;
}

// RaBitQ constants
inline constexpr float EPSILON_0 = 1.9f;  // Confidence parameter (Section 5.2.4)
inline constexpr int B_QUERY = 4;          // Query quantization bits (Theorem 3.3)

}  // namespace rabitq
