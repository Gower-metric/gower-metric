// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "../utils/arith.hpp"
#include "../utils/enums.hpp"

template <typename T>
struct CategoricalOrdinalParams {
  const T* x;
  const T* y;
  int i;
  const double* cum;
  std::size_t levels;
  T denom;
  double podani_denom;
  CategoricalOrdinalCalculationTypes mode;
};

template <typename T>
auto compute_categorical_ordinal_diff(const CategoricalOrdinalParams<T>& params)
    -> T {
  const int i = params.i;
  const T rx = params.x[i];
  const T ry = params.y[i];

  switch (params.mode) {
    case CategoricalOrdinalCalculationTypes::PODANI: {
      if (rx == ry) return T(0);
      const std::size_t ix = gower_detail::level_index(rx, params.levels);
      const std::size_t iy = gower_detail::level_index(ry, params.levels);
      if (ix == params.levels || iy == params.levels) return T(1);
      const bool x_lower = rx < ry;
      const std::size_t a = x_lower ? ix : iy;
      const std::size_t b = x_lower ? iy : ix;
      const double d =
          (1.0 + (params.cum[b] - params.cum[a + 1])) / params.podani_denom;
      return static_cast<T>(d < 1.0 ? d : 1.0);
    }
    case CategoricalOrdinalCalculationTypes::KAUFMAN:
    default: {
      const T diff = gower_detail::abs_diff(rx, ry);
      const T d = static_cast<T>(diff / params.denom);
      return d < T(1) ? d : T(1);
    }
  }
}

// EOF
