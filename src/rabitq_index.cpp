#include "rabitq_index.hpp"
#include <cmath>

namespace rabitq {

Eigen::VectorXf compute_centroid(const Eigen::MatrixXf& X) {
    return X.colwise().mean();
}

IndexResult build_index(const Eigen::MatrixXf& X, uint64_t seed) {
    const int N = X.rows();
    const uint32_t D = X.cols();
    const uint32_t B = pad_dim(D);

    IndexResult result;
    result.D = D;
    result.B = B;

    // Step 1: Compute centroid
    result.centroid = compute_centroid(X);

    // Step 2: Generate random orthogonal matrix P [B × B]
    result.P = generate_orthogonal(B, seed);

    // Step 3: Pad data and centroid to B dimensions (zero-pad)
    Eigen::MatrixXf X_pad = Eigen::MatrixXf::Zero(N, B);
    X_pad.leftCols(D) = X;

    Eigen::VectorXf c_pad = Eigen::VectorXf::Zero(B);
    c_pad.head(D) = result.centroid;

    // Step 4: Inverse-rotate centered vectors: v = (o_r - c) @ P^T
    // Subtract centroid from each row
    Eigen::MatrixXf centered = X_pad.rowwise() - c_pad.transpose();
    Eigen::MatrixXf XP = inverse_rotate(centered, result.P);

    // Step 5: For each vector, extract binary code and precompute scalars
    const uint32_t words = words_per_code(B);
    const float sqrt_B = std::sqrt(static_cast<float>(B));

    result.vectors.resize(N);
    for (int i = 0; i < N; i++) {
        auto& qv = result.vectors[i];
        qv.code.resize(words, 0);

        // Compute dist_to_centroid = ||o_r - c|| (using original D dimensions)
        float norm = centered.row(i).head(D).norm();
        qv.dist_to_centroid = norm;

        // Extract signs and pack into uint64_t words
        // Also compute L1 norm of the rotated vector (for x0)
        float l1_norm = 0.0f;
        uint32_t pop = 0;
        for (uint32_t j = 0; j < B; j++) {
            float v_j = XP(i, j);
            l1_norm += std::abs(v_j);

            if (v_j > 0.0f) {
                // Set bit j: word index = j/64, bit position = j%64
                qv.code[j / 64] |= (1ULL << (j % 64));
                pop++;
            }
        }
        qv.popcount = pop;

        // Compute x0 = ⟨ō, o⟩ = ||v||_1 / √B, normalized by ||o_r - c||
        // v = P^{-1}(o_r - c) has norm ||o_r - c|| (rotation preserves norms)
        // The unit-normalized version is v / ||v|| = v / ||o_r - c||
        // So ⟨ō, o⟩ = (1/√B) · Σ|v_i| / ||v|| = l1_norm / (√B · norm)
        if (norm > 1e-10f) {
            qv.x0 = l1_norm / (sqrt_B * norm);
        } else {
            // Degenerate case: vector is at centroid. Set x0 = 0.8 (safe default).
            // Reference: rabitq.py sets x0[~np.isfinite(x0)] = 0.8
            qv.x0 = 0.8f;
        }
    }

    return result;
}

}  // namespace rabitq
