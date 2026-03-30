#include "rotation.hpp"

namespace rabitq {

Eigen::MatrixXf generate_orthogonal(uint32_t D, uint64_t seed) {
    std::mt19937_64 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Fill D×D matrix with i.i.d. N(0,1) entries
    Eigen::MatrixXf G(D, D);
    for (uint32_t i = 0; i < D * D; i++) {
        G.data()[i] = dist(rng);
    }

    // QR decomposition: G = Q R, where Q is orthogonal
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(G);
    Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(D, D);

    return Q;
}

Eigen::MatrixXf inverse_rotate(const Eigen::MatrixXf& X, const Eigen::MatrixXf& P) {
    return X * P.transpose();
}

}  // namespace rabitq
