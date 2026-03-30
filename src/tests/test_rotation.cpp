#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include "rotation.hpp"

using namespace rabitq;

// ---------------------------------------------------------------------------
// Test: P^T P = I (orthogonality) for various dimensions
// ---------------------------------------------------------------------------
TEST(Rotation, Orthogonality) {
    for (uint32_t D : {64, 128, 256, 512, 960}) {
        auto P = generate_orthogonal(D, 42);

        ASSERT_EQ(P.rows(), D);
        ASSERT_EQ(P.cols(), D);

        Eigen::MatrixXf PtP = P.transpose() * P;
        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(D, D);

        float max_err = (PtP - I).cwiseAbs().maxCoeff();
        EXPECT_LT(max_err, 1e-5f)
            << "D=" << D << ": P^T P should be identity, max error: " << max_err;
    }
}

// ---------------------------------------------------------------------------
// Test: |det(P)| = 1 (orthogonal, not just orthonormal columns)
// ---------------------------------------------------------------------------
TEST(Rotation, Determinant) {
    for (uint32_t D : {64, 128, 256}) {
        auto P = generate_orthogonal(D, 123);
        float det = P.determinant();
        EXPECT_NEAR(std::abs(det), 1.0f, 1e-3f)
            << "D=" << D << ": |det(P)| should be 1, got: " << det;
    }
}

// ---------------------------------------------------------------------------
// Test: Different seeds produce different matrices
// ---------------------------------------------------------------------------
TEST(Rotation, DifferentSeeds) {
    auto P1 = generate_orthogonal(128, 1);
    auto P2 = generate_orthogonal(128, 2);

    float diff = (P1 - P2).cwiseAbs().maxCoeff();
    EXPECT_GT(diff, 0.01f) << "Different seeds should produce different matrices";
}

// ---------------------------------------------------------------------------
// Test: Same seed produces identical matrices (reproducibility)
// ---------------------------------------------------------------------------
TEST(Rotation, Reproducibility) {
    auto P1 = generate_orthogonal(128, 42);
    auto P2 = generate_orthogonal(128, 42);

    float diff = (P1 - P2).cwiseAbs().maxCoeff();
    EXPECT_FLOAT_EQ(diff, 0.0f) << "Same seed should produce identical matrices";
}

// ---------------------------------------------------------------------------
// Test: inverse_rotate preserves norms (rotation is isometric)
// ---------------------------------------------------------------------------
TEST(Rotation, InverseRotatePreservesNorms) {
    constexpr uint32_t D = 128;
    constexpr int N = 100;

    auto P = generate_orthogonal(D, 42);

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::MatrixXf X(N, D);
    for (int i = 0; i < N * D; i++)
        X.data()[i] = dist(rng);

    Eigen::MatrixXf XP = inverse_rotate(X, P);

    ASSERT_EQ(XP.rows(), N);
    ASSERT_EQ(XP.cols(), D);

    for (int i = 0; i < N; i++) {
        float norm_before = X.row(i).norm();
        float norm_after = XP.row(i).norm();
        EXPECT_NEAR(norm_before, norm_after, 1e-3f)
            << "Row " << i << ": rotation should preserve norms";
    }
}

// ---------------------------------------------------------------------------
// Test: inverse_rotate preserves inner products
// ---------------------------------------------------------------------------
TEST(Rotation, InverseRotatePreservesInnerProducts) {
    constexpr uint32_t D = 128;

    auto P = generate_orthogonal(D, 42);

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::VectorXf a(D), b(D);
    for (uint32_t i = 0; i < D; i++) {
        a(i) = dist(rng);
        b(i) = dist(rng);
    }

    Eigen::MatrixXf AB(2, D);
    AB.row(0) = a.transpose();
    AB.row(1) = b.transpose();
    Eigen::MatrixXf ABP = inverse_rotate(AB, P);

    float ip_before = a.dot(b);
    float ip_after = ABP.row(0).dot(ABP.row(1));
    EXPECT_NEAR(ip_before, ip_after, 1e-3f)
        << "Rotation should preserve inner products";
}

// ---------------------------------------------------------------------------
// Benchmark: matrix generation time for typical dimensions
// ---------------------------------------------------------------------------
TEST(Rotation, BenchmarkGeneration) {
    std::cout << "  Rotation matrix generation benchmark:" << std::endl;
    for (uint32_t D : {128, 256, 512, 960}) {
        auto start = std::chrono::high_resolution_clock::now();
        auto P = generate_orthogonal(D, 42);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "    D=" << D << ": " << ms << " ms" << std::endl;

        // Sanity check
        EXPECT_EQ(P.rows(), D);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: inverse_rotate throughput
// ---------------------------------------------------------------------------
TEST(Rotation, BenchmarkInverseRotate) {
    constexpr int N = 10000;
    std::cout << "  Inverse rotation benchmark (" << N << " vectors):" << std::endl;

    for (uint32_t D : {128, 256, 512, 960}) {
        auto P = generate_orthogonal(D, 42);

        std::mt19937 rng(99);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        Eigen::MatrixXf X(N, D);
        for (int i = 0; i < N * (int)D; i++)
            X.data()[i] = dist(rng);

        auto start = std::chrono::high_resolution_clock::now();
        auto XP = inverse_rotate(X, P);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "    D=" << D << ": " << ms << " ms"
                  << " (" << (N / ms * 1000) << " vec/s)" << std::endl;

        EXPECT_EQ(XP.rows(), N);
    }
}
