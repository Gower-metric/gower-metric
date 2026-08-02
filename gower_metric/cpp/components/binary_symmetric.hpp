// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

template <typename T>
struct BinarySymmetricParams {
  const T* x;
  const T* y;
  int i;
};

template <typename T>
auto compute_binary_symmetric_diff(const BinarySymmetricParams<T>& params)
    -> T {
  const int i = params.i;
  const bool x_positive = params.x[i] == T(1);
  const bool y_positive = params.y[i] == T(1);

  return x_positive == y_positive ? T(0) : T(1);
}

// EOF
