#include <gtest/gtest.h>
#include <immintrin.h>
#include <chrono>
#include <random>
#include "defines.hpp"

using namespace rabitq;

// ---------------------------------------------------------------------------
// Test 1: Eigen matrix multiplication works correctly
// ---------------------------------------------------------------------------
TEST(Setup, EigenMatMul) {
    // 4×3 times 3×2 = 4×2
    Eigen::MatrixXf A(4, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12;

    Eigen::MatrixXf B(3, 2);
    B << 1, 0,
         0, 1,
         1, 1;

    Eigen::MatrixXf C = A * B;

    EXPECT_EQ(C.rows(), 4);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_FLOAT_EQ(C(0, 0), 4.0f);   // 1*1 + 2*0 + 3*1
    EXPECT_FLOAT_EQ(C(0, 1), 5.0f);   // 1*0 + 2*1 + 3*1
    EXPECT_FLOAT_EQ(C(3, 0), 22.0f);  // 10*1 + 11*0 + 12*1
    EXPECT_FLOAT_EQ(C(3, 1), 23.0f);  // 10*0 + 11*1 + 12*1
}

// ---------------------------------------------------------------------------
// Test 2: QR decomposition produces orthogonal matrix
// ---------------------------------------------------------------------------
TEST(Setup, QROrthogonal) {
    constexpr int D = 128;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random D×D matrix with i.i.d. N(0,1) entries
    Eigen::MatrixXf G(D, D);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            G(i, j) = dist(rng);

    // QR decomposition → Q is orthogonal
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(G);
    Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(D, D);

    // Verify Q^T Q ≈ I (within 1e-5)
    Eigen::MatrixXf QtQ = Q.transpose() * Q;
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(D, D);

    float max_err = (QtQ - I).cwiseAbs().maxCoeff();
    EXPECT_LT(max_err, 1e-5f) << "Q^T Q should be identity, max error: " << max_err;

    // Verify |det(Q)| = 1 (orthogonal)
    float det = Q.determinant();
    EXPECT_NEAR(std::abs(det), 1.0f, 1e-4f) << "det(Q) should be ±1, got: " << det;
}

// ---------------------------------------------------------------------------
// Test 3: AVX2 popcount works
// ---------------------------------------------------------------------------
TEST(Setup, AVX2Popcount) {
    // Test __builtin_popcountll
    uint64_t x = 0xFFFF'FFFF'FFFF'FFFFull;  // 64 ones
    EXPECT_EQ(__builtin_popcountll(x), 64);

    x = 0xAAAA'AAAA'AAAA'AAAAull;  // alternating 10 pattern → 32 ones
    EXPECT_EQ(__builtin_popcountll(x), 32);

    x = 0;
    EXPECT_EQ(__builtin_popcountll(x), 0);

    x = 1;
    EXPECT_EQ(__builtin_popcountll(x), 1);
}

// ---------------------------------------------------------------------------
// Test 4: AVX2 SIMD instructions are functional
// ---------------------------------------------------------------------------
TEST(Setup, AVX2Functional) {
    // Load 8 floats into a 256-bit register
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    __m256 v = _mm256_loadu_ps(data);

    // Add the vector to itself
    __m256 result = _mm256_add_ps(v, v);

    float out[8];
    _mm256_storeu_ps(out, result);

    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(out[i], data[i] * 2.0f);
    }
}

// ---------------------------------------------------------------------------
// Test 5: defines.hpp constants and utilities
// ---------------------------------------------------------------------------
TEST(Setup, DefinesConstants) {
    EXPECT_EQ(words_per_code(128), 2u);
    EXPECT_EQ(words_per_code(960), 15u);
    EXPECT_EQ(words_per_code(96), 2u);
    EXPECT_EQ(words_per_code(64), 1u);
    EXPECT_EQ(words_per_code(65), 2u);

    EXPECT_EQ(pad_dim(128), 128u);
    EXPECT_EQ(pad_dim(96), 128u);
    EXPECT_EQ(pad_dim(960), 960u);
    EXPECT_EQ(pad_dim(100), 128u);

    EXPECT_FLOAT_EQ(EPSILON_0, 1.9f);
    EXPECT_EQ(B_QUERY, 4);
}

// ---------------------------------------------------------------------------
// Test 6: Benchmark matrix multiplication for typical RaBitQ dimensions
// ---------------------------------------------------------------------------
TEST(Setup, BenchmarkMatMul) {
    constexpr int N = 1000;  // Number of vectors

    for (int D : {128, 256, 512, 960}) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Generate random orthogonal matrix P (D×D)
        Eigen::MatrixXf G(D, D);
        for (int i = 0; i < D * D; i++)
            G.data()[i] = dist(rng);
        Eigen::HouseholderQR<Eigen::MatrixXf> qr(G);
        Eigen::MatrixXf P = qr.householderQ() * Eigen::MatrixXf::Identity(D, D);

        // Generate N random vectors
        Eigen::MatrixXf X(N, D);
        for (int i = 0; i < N * D; i++)
            X.data()[i] = dist(rng);

        // Benchmark: X × P^T (inverse-transform N vectors)
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXf XP = X * P.transpose();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  D=" << D << ": " << N << " vectors × P^T in " << ms << " ms"
                  << " (" << (N / ms * 1000) << " vec/s)" << std::endl;

        // Sanity: rotation preserves norms (within tolerance)
        float norm_before = X.row(0).norm();
        float norm_after = XP.row(0).norm();
        EXPECT_NEAR(norm_before, norm_after, 1e-3f)
            << "Rotation should preserve norms for D=" << D;
    }
}
