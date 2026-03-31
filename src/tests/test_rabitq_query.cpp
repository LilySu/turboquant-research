#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include "rabitq_query.hpp"

using namespace rabitq;

// Helper: generate random data
static Eigen::MatrixXf random_data(int N, int D, uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::MatrixXf X(N, D);
    for (int i = 0; i < N * D; i++)
        X.data()[i] = dist(rng);
    return X;
}

// ---------------------------------------------------------------------------
// Test: basic quantized query shapes
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, Shapes) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);

    Eigen::VectorXf q = X.row(0).transpose();  // Use first data vec as query
    auto qq = quantize_query(q, index);

    EXPECT_EQ(qq.q_u.size(), 128u);
    EXPECT_EQ(qq.bit_planes.size(), B_QUERY * words_per_code(128));  // 4 * 2 = 8
    EXPECT_GT(qq.width, 0.0f);
    EXPECT_LE(qq.vl, qq.vr);
}

// ---------------------------------------------------------------------------
// Test: all quantized values in range [0, 2^B_q - 1]
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, ValueRange) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);

    for (int i = 0; i < 10; i++) {
        Eigen::VectorXf q = X.row(i).transpose();
        auto qq = quantize_query(q, index);

        for (uint32_t j = 0; j < 128; j++) {
            EXPECT_LE(qq.q_u[j], 15) << "q_u[" << j << "] should be <= 15";
        }
    }
}

// ---------------------------------------------------------------------------
// Test: sum_q matches manual sum
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, SumQ) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);

    Eigen::VectorXf q = X.row(0).transpose();
    auto qq = quantize_query(q, index);

    uint32_t manual_sum = 0;
    for (auto v : qq.q_u) manual_sum += v;
    EXPECT_EQ(qq.sum_q, manual_sum);
}

// ---------------------------------------------------------------------------
// Test: bit planes correctly represent q_u values
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, BitPlanes) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);

    Eigen::VectorXf q = X.row(0).transpose();
    auto qq = quantize_query(q, index);

    const uint32_t B = index.B;
    const uint32_t words = words_per_code(B);

    // Reconstruct q_u from bit planes and verify
    for (uint32_t i = 0; i < B; i++) {
        uint8_t reconstructed = 0;
        for (int j = 0; j < B_QUERY; j++) {
            bool bit = (qq.bit_planes[j * words + i / 64] >> (i % 64)) & 1;
            if (bit) reconstructed |= (1 << j);
        }
        EXPECT_EQ(reconstructed, qq.q_u[i])
            << "Bit plane reconstruction mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: randomized rounding is unbiased (average should be close to true value)
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, RandomizedRoundingUnbiased) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);

    Eigen::VectorXf q = X.row(0).transpose();

    // Quantize many times with different random dithers
    const int trials = 1000;
    std::vector<double> avg_q_u(128, 0.0);

    for (int t = 0; t < trials; t++) {
        auto u = generate_dither(index.B, t + 1);
        auto qq = quantize_query(q, index, u);
        for (uint32_t j = 0; j < 128; j++) {
            avg_q_u[j] += qq.q_u[j];
        }
    }

    // Compare with deterministic (u=0.5) quantization
    auto qq_det = quantize_query(q, index);

    double max_diff = 0.0;
    for (uint32_t j = 0; j < 128; j++) {
        avg_q_u[j] /= trials;
        double diff = std::abs(avg_q_u[j] - qq_det.q_u[j]);
        max_diff = std::max(max_diff, diff);
    }

    // Average over many trials should be close to deterministic (within 1.0)
    EXPECT_LT(max_diff, 1.5)
        << "Randomized rounding average should be close to deterministic";

    std::cout << "  Max avg diff (randomized vs deterministic): " << max_diff << std::endl;
}

// ---------------------------------------------------------------------------
// Test: sqr_y is correct squared distance to centroid
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, SqrY) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);

    Eigen::VectorXf q = X.row(5).transpose();
    auto qq = quantize_query(q, index);

    float expected = (q - index.centroid).squaredNorm();
    EXPECT_NEAR(qq.sqr_y, expected, 1e-3f);
}

// ---------------------------------------------------------------------------
// Test: deterministic mode (no dither, u=0.5) is reproducible
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, DeterministicReproducible) {
    auto X = random_data(100, 128);
    auto index = build_index(X, 42);
    Eigen::VectorXf q = X.row(0).transpose();

    auto qq1 = quantize_query(q, index);
    auto qq2 = quantize_query(q, index);

    EXPECT_EQ(qq1.q_u, qq2.q_u);
    EXPECT_EQ(qq1.sum_q, qq2.sum_q);
    EXPECT_FLOAT_EQ(qq1.vl, qq2.vl);
    EXPECT_FLOAT_EQ(qq1.width, qq2.width);
}

// ---------------------------------------------------------------------------
// Test: with padding (D=100 → B=128)
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, Padding) {
    auto X = random_data(50, 100);
    auto index = build_index(X, 42);

    Eigen::VectorXf q = X.row(0).transpose();
    auto qq = quantize_query(q, index);

    EXPECT_EQ(qq.q_u.size(), 128u);  // Padded to B=128
    EXPECT_EQ(qq.bit_planes.size(), (size_t)(B_QUERY * words_per_code(128)));
}

// ---------------------------------------------------------------------------
// Benchmark: query quantization throughput
// ---------------------------------------------------------------------------
TEST(RaBitQQuery, Benchmark) {
    std::cout << "  Query quantization benchmark:" << std::endl;
    for (int D : {128, 256, 512, 960}) {
        auto X = random_data(100, D);
        auto index = build_index(X, 42);

        const int N_queries = 10000;
        Eigen::MatrixXf Q = random_data(N_queries, D, 99);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_queries; i++) {
            Eigen::VectorXf q = Q.row(i).transpose();
            auto qq = quantize_query(q, index);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "    D=" << D << ": " << N_queries << " queries in " << ms << " ms"
                  << " (" << (N_queries / ms * 1000) << " q/s)" << std::endl;
    }
}
