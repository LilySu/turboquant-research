#pragma once

#include "defines.hpp"
#include <random>

namespace rabitq {

/// Generate a random orthogonal matrix P of size D×D via QR decomposition
/// of a matrix with i.i.d. N(0,1) entries.
///
/// The resulting matrix satisfies:
///   - P^T P = P P^T = I  (orthogonal)
///   - |det(P)| = 1
///
/// This is used for:
///   - Index phase: P^{-1} = P^T applied to data vectors
///   - Query phase: P^{-1} = P^T applied to query vectors
///
/// @param D     Dimension (typically pad_dim of original dimension)
/// @param seed  Random seed for reproducibility (0 = random device)
/// @return      D×D orthogonal matrix
Eigen::MatrixXf generate_orthogonal(uint32_t D, uint64_t seed = 0);

/// Apply inverse rotation P^{-1} = P^T to a batch of row vectors.
///
/// Given X of shape [N, D] and P of shape [D, D], computes X * P^T.
/// Each row of X is a data or query vector; the result is the
/// inverse-transformed vectors in codebook coordinate space.
///
/// @param X  Matrix of row vectors [N, D]
/// @param P  Orthogonal matrix [D, D]
/// @return   X * P^T [N, D]
Eigen::MatrixXf inverse_rotate(const Eigen::MatrixXf& X, const Eigen::MatrixXf& P);

}  // namespace rabitq
