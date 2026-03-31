#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include "rabitq_index.hpp"

using namespace rabitq;

// ---------------------------------------------------------------------------
// Helper: generate random data matrix
// ---------------------------------------------------------------------------
static Eigen::MatrixXf random_data(int N, int D, uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::MatrixXf X(N, D);
    for (int i = 0; i < N * D; i++)
        X.data()[i] = dist(rng);
    return X;
}

// ---------------------------------------------------------------------------
// Test: centroid computation
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, Centroid) {
    Eigen::MatrixXf X(3, 4);
    X << 1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12;
    auto c = compute_centroid(X);
    EXPECT_FLOAT_EQ(c(0), 5.0f);
    EXPECT_FLOAT_EQ(c(1), 6.0f);
    EXPECT_FLOAT_EQ(c(2), 7.0f);
    EXPECT_FLOAT_EQ(c(3), 8.0f);
}

// ---------------------------------------------------------------------------
// Test: build_index produces correct shapes
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, Shapes) {
    auto X = random_data(100, 128);
    auto result = build_index(X, 42);

    EXPECT_EQ(result.D, 128u);
    EXPECT_EQ(result.B, 128u);  // 128 is already a multiple of 64
    EXPECT_EQ(result.centroid.size(), 128);
    EXPECT_EQ(result.P.rows(), 128);
    EXPECT_EQ(result.P.cols(), 128);
    EXPECT_EQ(result.vectors.size(), 100u);

    // Each code should have B/64 = 2 uint64_t words
    for (const auto& qv : result.vectors) {
        EXPECT_EQ(qv.code.size(), 2u);
    }
}

// ---------------------------------------------------------------------------
// Test: build_index with dimension needing padding (D=100 → B=128)
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, Padding) {
    auto X = random_data(50, 100);
    auto result = build_index(X, 42);

    EXPECT_EQ(result.D, 100u);
    EXPECT_EQ(result.B, 128u);  // Padded to next multiple of 64
    EXPECT_EQ(result.P.rows(), 128);
    EXPECT_EQ(result.P.cols(), 128);
    EXPECT_EQ(result.vectors[0].code.size(), 2u);  // 128/64 = 2
}

// ---------------------------------------------------------------------------
// Test: x0 (⟨ō,o⟩) averages around 0.8 for D=128
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, X0Average) {
    auto X = random_data(1000, 128);
    auto result = build_index(X, 42);

    double sum_x0 = 0.0;
    for (const auto& qv : result.vectors) {
        sum_x0 += qv.x0;
        // x0 should be positive
        EXPECT_GT(qv.x0, 0.0f) << "x0 should always be positive";
        // x0 should be less than 1 (can't have perfect quantization)
        EXPECT_LT(qv.x0, 1.0f) << "x0 should be less than 1";
    }
    double avg_x0 = sum_x0 / result.vectors.size();

    // Expected: ~0.798 for D=128 (from paper)
    EXPECT_NEAR(avg_x0, 0.798, 0.02)
        << "Average x0 should be ~0.798 for D=128, got: " << avg_x0;

    std::cout << "  D=128: avg ⟨ō,o⟩ = " << avg_x0
              << " (expected ~0.798)" << std::endl;
}

// ---------------------------------------------------------------------------
// Test: x0 for higher dimensions (should still be ~0.798-0.800)
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, X0HigherDimensions) {
    for (int D : {256, 512}) {
        auto X = random_data(500, D);
        auto result = build_index(X, 42);

        double sum_x0 = 0.0;
        for (const auto& qv : result.vectors)
            sum_x0 += qv.x0;
        double avg_x0 = sum_x0 / result.vectors.size();

        EXPECT_NEAR(avg_x0, 0.798, 0.02)
            << "D=" << D << ": avg x0 should be ~0.798, got: " << avg_x0;

        std::cout << "  D=" << D << ": avg ⟨ō,o⟩ = " << avg_x0 << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Test: dist_to_centroid is correct
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, DistToCentroid) {
    auto X = random_data(100, 128);
    auto result = build_index(X, 42);

    // Manually compute distances
    for (int i = 0; i < 100; i++) {
        float expected = (X.row(i).transpose() - result.centroid).norm();
        EXPECT_NEAR(result.vectors[i].dist_to_centroid, expected, 1e-4f)
            << "Vector " << i << " dist_to_centroid mismatch";
    }
}

// ---------------------------------------------------------------------------
// Test: binary code reconstruction gives correct quantized vector
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, CodeReconstruction) {
    auto X = random_data(10, 128);
    auto result = build_index(X, 42);
    const uint32_t B = result.B;
    const float sqrt_B = std::sqrt(static_cast<float>(B));

    for (int i = 0; i < 10; i++) {
        const auto& qv = result.vectors[i];

        // Reconstruct x̄ = (2·x̄_b - 1) / √B
        Eigen::VectorXf x_bar(B);
        for (uint32_t j = 0; j < B; j++) {
            bool bit = (qv.code[j / 64] >> (j % 64)) & 1;
            x_bar(j) = bit ? (1.0f / sqrt_B) : (-1.0f / sqrt_B);
        }

        // Verify x̄ is a unit vector
        EXPECT_NEAR(x_bar.norm(), 1.0f, 1e-5f)
            << "Reconstructed x̄ should be unit vector";

        // Verify popcount matches
        uint32_t manual_pop = 0;
        for (uint32_t j = 0; j < B; j++) {
            if ((qv.code[j / 64] >> (j % 64)) & 1) manual_pop++;
        }
        EXPECT_EQ(qv.popcount, manual_pop);
    }
}

// ---------------------------------------------------------------------------
// Test: x0 matches manual computation via code reconstruction
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, X0MatchesReconstruction) {
    auto X = random_data(10, 128);
    auto result = build_index(X, 42);
    const uint32_t B = result.B;
    const float sqrt_B = std::sqrt(static_cast<float>(B));

    for (int i = 0; i < 10; i++) {
        const auto& qv = result.vectors[i];

        // Reconstruct x̄ from bits
        Eigen::VectorXf x_bar(B);
        for (uint32_t j = 0; j < B; j++) {
            bool bit = (qv.code[j / 64] >> (j % 64)) & 1;
            x_bar(j) = bit ? (1.0f / sqrt_B) : (-1.0f / sqrt_B);
        }

        // Compute ō = P^T · x̄  (P = Q from QR; effective rotation is Q^T)
        // Since inverse_rotate computes X @ Q^T, the forward rotation is Q^T.
        Eigen::VectorXf o_bar = result.P.transpose() * x_bar;

        // Compute unit vector o = (o_r - c) / ||o_r - c||
        Eigen::VectorXf centered = X.row(i).transpose() - result.centroid;
        float norm = centered.norm();
        // Pad centered to B dims
        Eigen::VectorXf centered_pad = Eigen::VectorXf::Zero(B);
        centered_pad.head(result.D) = centered;
        Eigen::VectorXf o_unit = centered_pad / norm;

        // ⟨ō, o⟩ should match x0
        float manual_x0 = o_bar.dot(o_unit);
        EXPECT_NEAR(qv.x0, manual_x0, 1e-3f)
            << "Vector " << i << ": x0 mismatch";
    }
}

// ---------------------------------------------------------------------------
// Test: reproducibility (same seed → same result)
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, Reproducibility) {
    auto X = random_data(50, 128);
    auto r1 = build_index(X, 42);
    auto r2 = build_index(X, 42);

    for (int i = 0; i < 50; i++) {
        EXPECT_EQ(r1.vectors[i].code, r2.vectors[i].code);
        EXPECT_FLOAT_EQ(r1.vectors[i].x0, r2.vectors[i].x0);
        EXPECT_FLOAT_EQ(r1.vectors[i].dist_to_centroid, r2.vectors[i].dist_to_centroid);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: indexing throughput
// ---------------------------------------------------------------------------
TEST(RaBitQIndex, BenchmarkIndexing) {
    std::cout << "  Indexing benchmark:" << std::endl;
    for (int D : {128, 256, 512, 960}) {
        const int N = 10000;
        auto X = random_data(N, D);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = build_index(X, 42);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "    D=" << D << ": " << N << " vectors in " << ms << " ms"
                  << " (" << (N / ms * 1000) << " vec/s)" << std::endl;

        EXPECT_EQ(result.vectors.size(), (size_t)N);
    }
}
