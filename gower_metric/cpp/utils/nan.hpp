#pragma once

#include <cmath>
#include <type_traits>

namespace gower_detail {

template <typename T>
inline auto is_present(const T v) -> bool {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return !std::isnan(v);
  } else {
    return !std::isnan(static_cast<double>(v));
  }
}

template <typename T>
inline auto is_missing(const T a, const T b) -> bool {
  return !is_present(a) || !is_present(b);
}

}  // namespace gower_detail

// EOF
