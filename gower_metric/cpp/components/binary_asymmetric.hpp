// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

template <typename T>
struct BinaryAsymmetricParams {
  const T* x;
  const T* y;
  int i;
};

template <typename T>
struct BinaryAsymmetricDiff {
  T diff;
  bool present;
};

template <typename T>
auto compute_binary_asymmetric_diff(const BinaryAsymmetricParams<T>& params)
    -> BinaryAsymmetricDiff<T> {
  const int i = params.i;
  const bool x_positive = params.x[i] == T(1);
  const bool y_positive = params.y[i] == T(1);

  const bool present = x_positive || y_positive;

  return {x_positive != y_positive ? T(1) : T(0), present};
}

// EOF
