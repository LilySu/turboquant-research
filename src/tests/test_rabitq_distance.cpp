#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include "rabitq_distance.hpp"

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
// Test: ip_byte_bin correctness
// ---------------------------------------------------------------------------
TEST(RaBitQDistance, IpByteBin) {
    // Manual example: B=64 (1 word), B_q=4
    // code = all 1s: 0xFFFFFFFFFFFFFFFF
    // query q_u = all 15s → bit planes all 1s
    uint64_t code[1] = {0xFFFFFFFFFFFFFFFFULL};
    uint64_t bit_planes[4] = {
        0xFFFFFFFFFFFFFFFFULL,  // plane 0
        0xFFFFFFFFFFFFFFFFULL,  // plane 1
        0xFFFFFFFFFFFFFFFFULL,  // plane 2
        0xFFFFFFFFFFFFFFFFULL,  // plane 3
    };

    // ⟨x̄_b, q̄_u⟩ = Σ 2^j · popcount(all_ones & all_ones) = (1+2+4+8) × 64 = 960
    uint32_t result = ip_byte_bin(code, bit_planes, 1);
    EXPECT_EQ(result, 15u * 64u);  // 960
}

TEST(RaBitQDistance, IpByteBinPartial) {
    // B=64, code = alternating: 0xAAAA...A (32 ones)
    // All bit planes = 0xFFFF...F (all ones)
    uint64_t code[1] = {0xAAAAAAAAAAAAAAAAULL};
    uint64_t bit_planes[4] = {
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
    };

    // popcount(code & plane) = 32 for each plane
    // result = (1+2+4+8) × 32 = 480
    uint32_t result = ip_byte_bin(code, bit_planes, 1);
    EXPECT_EQ(result, 15u * 32u);  // 480
}

// ---------------------------------------------------------------------------
// Test: unbiasedness — average estimated IP ≈ true IP over many rotations
// ---------------------------------------------------------------------------
TEST(RaBitQDistance, Unbiasedness) {
    constexpr int D = 128;
    constexpr int N_data = 200;
    constexpr int N_trials = 50;  // Different random rotations

    // Fixed data and query
    auto X = random_data(N_data, D, 42);
    Eigen::VectorXf query = random_data(1, D, 99).row(0).transpose();

    // Compute true distances
    std::vector<float> true_dists(N_data);
    for (int i = 0; i < N_data; i++) {
        true_dists[i] = (X.row(i).transpose() - query).squaredNorm();
    }

    // Average estimated distances over many random rotations
    std::vector<double> sum_estimated(N_data, 0.0);
    for (int trial = 0; trial < N_trials; trial++) {
        auto index = build_index(X, trial + 1);
        auto qq = quantize_query(query, index);

        for (int i = 0; i < N_data; i++) {
            auto est = estimate_distance(index.vectors[i], qq, index.B);
            sum_estimated[i] += est.estimated_dist;
        }
    }

    // Check average error is near zero
    double total_error = 0.0;
    double total_abs_error = 0.0;
    for (int i = 0; i < N_data; i++) {
        double avg = sum_estimated[i] / N_trials;
        double error = avg - true_dists[i];
        total_error += error;
        total_abs_error += std::abs(error);
    }
    double mean_error = total_error / N_data;
    double mean_abs_error = total_abs_error / N_data;

    std::cout << "  D=" << D << ", N_trials=" << N_trials << std::endl;
    std::cout << "  Mean error (should be ~0): " << mean_error << std::endl;
    std::cout << "  Mean |error|: " << mean_abs_error << std::endl;

    // Mean error should be close to 0 (unbiased)
    // Allow some slack since we have finite samples
    double avg_true_dist = 0;
    for (int i = 0; i < N_data; i++) avg_true_dist += true_dists[i];
    avg_true_dist /= N_data;

    EXPECT_LT(std::abs(mean_error), avg_true_dist * 0.1)
        << "Mean error should be < 10% of mean true distance";
}

// ---------------------------------------------------------------------------
// Test: error bound holds with high probability
// ---------------------------------------------------------------------------
TEST(RaBitQDistance, ErrorBoundHolds) {
    constexpr int D = 128;
    constexpr int N_data = 500;

    auto X = random_data(N_data, D, 42);
    auto index = build_index(X, 42);

    Eigen::VectorXf query = random_data(1, D, 99).row(0).transpose();
    auto qq = quantize_query(query, index);

    int violations = 0;
    for (int i = 0; i < N_data; i++) {
        auto est = estimate_distance(index.vectors[i], qq, index.B);

        // True distance
        float true_dist = (X.row(i).transpose() - query).squaredNorm();

        // Lower bound should be ≤ true distance (with high probability)
        if (est.lower_bound_dist > true_dist) {
            violations++;
        }
    }

    double violation_rate = static_cast<double>(violations) / N_data;
    std::cout << "  Error bound violations: " << violations << "/" << N_data
              << " (" << violation_rate * 100 << "%)" << std::endl;

    // With ε₀=1.9, failure probability ≈ 2·exp(-c₀·1.9²) — should be very small
    EXPECT_LT(violation_rate, 0.10)
        << "Error bound should hold for ≥90% of vectors";
}

// ---------------------------------------------------------------------------
// Test: lower bound is always ≤ estimated distance
// ---------------------------------------------------------------------------
TEST(RaBitQDistance, LowerBoundConsistency) {
    constexpr int D = 128;
    auto X = random_data(100, D, 42);
    auto index = build_index(X, 42);

    Eigen::VectorXf query = random_data(1, D, 99).row(0).transpose();
    auto qq = quantize_query(query, index);

    for (int i = 0; i < 100; i++) {
        auto est = estimate_distance(index.vectors[i], qq, index.B);
        EXPECT_LE(est.lower_bound_dist, est.estimated_dist + 1e-6f)
            << "Lower bound should be ≤ estimated distance";
        EXPECT_GT(est.error_bound, 0.0f)
            << "Error bound should be positive";
    }
}

// ---------------------------------------------------------------------------
// Test: distance estimate is roughly correct (within 2x of true)
// ---------------------------------------------------------------------------
TEST(RaBitQDistance, RoughAccuracy) {
    constexpr int D = 128;
    auto X = random_data(100, D, 42);
    auto index = build_index(X, 42);

    Eigen::VectorXf query = random_data(1, D, 99).row(0).transpose();
    auto qq = quantize_query(query, index);

    int reasonable = 0;
    for (int i = 0; i < 100; i++) {
        auto est = estimate_distance(index.vectors[i], qq, index.B);
        float true_dist = (X.row(i).transpose() - query).squaredNorm();

        // Check if estimate is within 2x of true
        if (est.estimated_dist > 0 && true_dist > 0) {
            float ratio = est.estimated_dist / true_dist;
            if (ratio > 0.5f && ratio < 2.0f) reasonable++;
        }
    }

    EXPECT_GT(reasonable, 70)
        << "At least 70% of estimates should be within 2x of true distance";
    std::cout << "  Within 2x accuracy: " << reasonable << "/100" << std::endl;
}

// ---------------------------------------------------------------------------
// Benchmark: distance estimation throughput
// ---------------------------------------------------------------------------
TEST(RaBitQDistance, Benchmark) {
    std::cout << "  Distance estimation benchmark:" << std::endl;
    for (int D : {128, 256, 512}) {
        const int N = 10000;
        auto X = random_data(N, D);
        auto index = build_index(X, 42);

        Eigen::VectorXf query = random_data(1, D, 99).row(0).transpose();
        auto qq = quantize_query(query, index);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            auto est = estimate_distance(index.vectors[i], qq, index.B);
            (void)est;  // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "    D=" << D << ": " << N << " distances in " << ms << " ms"
                  << " (" << (N / ms * 1000) << " dist/s)" << std::endl;
    }
}
