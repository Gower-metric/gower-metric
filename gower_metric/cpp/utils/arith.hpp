// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace gower_detail {

template <typename T>
inline auto abs_diff(const T a, const T b) -> T {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return std::fabs(a - b);
  } else {
    const T d = a - b;
    return d < T(0) ? static_cast<T>(-d) : d;
  }
}

template <typename T>
inline auto level_index(const T code, const std::size_t levels) -> std::size_t {
  const auto c = static_cast<double>(code);
  const bool in_range = c >= 0.0 && c < static_cast<double>(levels);
  const auto idx = static_cast<std::size_t>(in_range ? c : 0.0);
  return (in_range && static_cast<double>(idx) == c) ? idx : levels;
}

}  // namespace gower_detail

// EOF
