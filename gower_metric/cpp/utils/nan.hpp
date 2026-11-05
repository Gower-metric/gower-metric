// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

namespace gower_detail {

template <typename T>
inline auto is_present(const T v) -> bool {
  return !(v != v);
}

template <typename T>
inline auto is_missing(const T a, const T b) -> bool {
  return !is_present(a) || !is_present(b);
}

}  // namespace gower_detail

// EOF
