// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include "../utils/arith.hpp"

template <typename T>
struct NumericParams {
  const T* x;
  const T* y;
  const T* ranges;
  const T* bandwidths;
  int i;
};

template <typename T>
auto compute_numeric_diff(const NumericParams<T>& params) -> T {
  const int i = params.i;
  const T abs_diff = gower_detail::abs_diff(params.x[i], params.y[i]);

  if (params.ranges[i] > 0) {
    T diff = static_cast<T>(abs_diff / params.ranges[i]);

    if (params.bandwidths[i] > 0 && abs_diff <= params.bandwidths[i]) {
      return T(0.0);
    }

    if (diff > T(1.0)) return T(1.0);
    return diff;
  }

  return abs_diff > T(0) ? T(1.0) : T(0.0);
}

// EOF
