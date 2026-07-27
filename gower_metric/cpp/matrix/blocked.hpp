#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "../config/gower_config.hpp"
#include "../utils/nan.hpp"

namespace gower_blocked {

constexpr std::size_t BLOCK = 128;

template <typename T>
auto supports_blocked(const CppConfig<T>& cfg) -> bool {
  if (cfg.conditional_distances) return false;
  if (cfg.missing_strategy == MissingStrategy::RAISE_ERROR) return false;
  return std::none_of(
      cfg.ordinal_group.begin(), cfg.ordinal_group.end(),
      [&cfg](const int i) -> bool { return cfg.ordinal_meta[i].use_podani; });
}

template <typename T>
inline auto has_missing(const T* data, const std::size_t count) -> bool {
  for (std::size_t i = 0; i < count; ++i) {
    if (!gower_detail::is_present(data[i])) return true;
  }
  return false;
}

template <typename T>
inline auto weight_total(const CppConfig<T>& cfg) -> T {
  const bool uniform = cfg.uniform_weights;
  const T* weights = cfg.feature_weights.data();
  const auto weight_of = [&](const int f) -> T {
    return uniform ? T(1) : weights[f];
  };

  T cat = 0;
  for (const int f : cfg.nominal_group) cat += weight_of(f);
  for (const int f : cfg.ordinal_group) cat += weight_of(f);
  for (const int f : cfg.binary_symmetric_group) cat += weight_of(f);
  T num = 0;
  for (const int f : cfg.numeric_group) num += weight_of(f);
  return cat + num;
}

template <typename T>
inline void transpose(const T* data, const std::size_t n, const std::size_t nf,
                      std::vector<T>& soa) {
  soa.resize(n * nf);
  for (std::size_t r = 0; r < n; ++r) {
    const T* row = data + r * nf;
    for (std::size_t f = 0; f < nf; ++f) {
      soa[f * n + r] = row[f];
    }
  }
}

template <typename T>
struct BlockParams {
  const T* x_soa;
  std::size_t x_stride;
  std::size_t xi;
  const T* y_soa;
  std::size_t y_stride;
  std::size_t j0;
  std::size_t len;
  T* out;
  bool similarity;
  T total_weight;
};

template <bool NanFree, bool ConstDenom, typename T>
inline void block(const CppConfig<T>& cfg, const BlockParams<T>& params) {
  static_assert(!ConstDenom || NanFree,
                "a constant denominator presumes no missing values");
  const T* x_soa = params.x_soa;
  const std::size_t x_stride = params.x_stride;
  const std::size_t xi = params.xi;
  const T* y_soa = params.y_soa;
  const std::size_t y_stride = params.y_stride;
  const std::size_t j0 = params.j0;
  const std::size_t len = params.len;
  T* out = params.out;
  const bool similarity = params.similarity;

  const bool uniform = cfg.uniform_weights;
  const bool max_dist = cfg.missing_strategy == MissingStrategy::MAX_DISTANCE;
  const T* weights = cfg.feature_weights.data();

  std::array<T, BLOCK> cat_diff{};
  std::array<T, BLOCK> num_diff{};
  std::array<T, BLOCK> cat_w{};
  std::array<T, BLOCK> num_w{};

  for (const int f : cfg.nominal_group) {
    const T xv = x_soa[static_cast<std::size_t>(f) * x_stride + xi];
    const T* col = y_soa + static_cast<std::size_t>(f) * y_stride + j0;
    const T wf = uniform ? T(1) : weights[f];
    const bool x_ok = gower_detail::is_present(xv);
    for (std::size_t b = 0; b < len; ++b) {
      const T yv = col[b];
      const bool ok = NanFree || (x_ok && gower_detail::is_present(yv));
      const T w = (ok || max_dist) ? wf : T(0);
      cat_diff[b] += w * (ok ? (xv == yv ? T(0) : T(1)) : T(1));
      if constexpr (!ConstDenom) cat_w[b] += w;
    }
  }

  for (const int f : cfg.ordinal_group) {
    const T xv = x_soa[static_cast<std::size_t>(f) * x_stride + xi];
    const T* col = y_soa + static_cast<std::size_t>(f) * y_stride + j0;
    const T wf = uniform ? T(1) : weights[f];
    const T denom = cfg.ordinal_meta[f].denom;
    const T denom_safe = denom > T(0) ? denom : T(1);
    const bool scaled = denom > T(0);
    const bool x_ok = gower_detail::is_present(xv);
    for (std::size_t b = 0; b < len; ++b) {
      const T yv = col[b];
      const bool ok = NanFree || (x_ok && gower_detail::is_present(yv));
      const T w = (ok || max_dist) ? wf : T(0);
      const T raw = xv < yv ? yv - xv : xv - yv;
      const T d = scaled ? raw / denom_safe : T(0);
      cat_diff[b] += w * (ok ? d : T(1));
      if constexpr (!ConstDenom) cat_w[b] += w;
    }
  }

  for (const int f : cfg.binary_symmetric_group) {
    const T xv = x_soa[static_cast<std::size_t>(f) * x_stride + xi];
    const T* col = y_soa + static_cast<std::size_t>(f) * y_stride + j0;
    const T wf = uniform ? T(1) : weights[f];
    const bool x_ok = gower_detail::is_present(xv);
    const bool x_pos = xv == T(1);
    for (std::size_t b = 0; b < len; ++b) {
      const T yv = col[b];
      const bool ok = NanFree || (x_ok && gower_detail::is_present(yv));
      const T w = (ok || max_dist) ? wf : T(0);
      cat_diff[b] += w * (ok ? (x_pos == (yv == T(1)) ? T(0) : T(1)) : T(1));
      if constexpr (!ConstDenom) cat_w[b] += w;
    }
  }

  for (const int f : cfg.binary_asymmetric_group) {
    const T xv = x_soa[static_cast<std::size_t>(f) * x_stride + xi];
    const T* col = y_soa + static_cast<std::size_t>(f) * y_stride + j0;
    const T wf = uniform ? T(1) : weights[f];
    const bool x_ok = gower_detail::is_present(xv);
    const bool x_pos = xv == T(1);
    for (std::size_t b = 0; b < len; ++b) {
      const T yv = col[b];
      const bool ok = NanFree || (x_ok && gower_detail::is_present(yv));
      const bool y_pos = yv == T(1);
      const bool counted = ok ? (x_pos || y_pos) : max_dist;
      const T w = counted ? wf : T(0);
      cat_diff[b] += w * (ok ? (x_pos != y_pos ? T(1) : T(0)) : T(1));
      if constexpr (!ConstDenom) cat_w[b] += w;
    }
  }

  for (const int f : cfg.numeric_group) {
    const T xv = x_soa[static_cast<std::size_t>(f) * x_stride + xi];
    const T* col = y_soa + static_cast<std::size_t>(f) * y_stride + j0;
    const T wf = uniform ? T(1) : weights[f];
    const T range = cfg.ranges[f];
    const T range_safe = range > T(0) ? range : T(1);
    const bool scaled = range > T(0);
    const T band = cfg.bandwidths[f];
    const bool x_ok = gower_detail::is_present(xv);
    for (std::size_t b = 0; b < len; ++b) {
      const T yv = col[b];
      const bool ok = NanFree || (x_ok && gower_detail::is_present(yv));
      const T w = (ok || max_dist) ? wf : T(0);
      const T raw = xv < yv ? yv - xv : xv - yv;
      T d = raw / range_safe;
      if (d > T(1)) d = T(1);
      if (band > T(0) && raw <= band) d = T(0);
      if (!scaled) d = T(0);
      num_diff[b] += w * (ok ? d : T(1));
      if constexpr (!ConstDenom) num_w[b] += w;
    }
  }

  const T nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  if constexpr (ConstDenom) {
    const T total_w = params.total_weight;
    for (std::size_t b = 0; b < len; ++b) {
      const T d = total_w > T(0) ? (cat_diff[b] + num_diff[b]) / total_w : nan;
      out[b] = similarity ? static_cast<T>(T(1) - d) : d;
    }
  } else {
    for (std::size_t b = 0; b < len; ++b) {
      const T total_w = cat_w[b] + num_w[b];
      const T d = total_w > T(0) ? (cat_diff[b] + num_diff[b]) / total_w : nan;
      out[b] = similarity ? static_cast<T>(T(1) - d) : d;
    }
  }
}

struct FastPath {
  bool nan_free = false;
  bool const_denom = false;
};

template <typename T>
inline void block_dispatch(const CppConfig<T>& cfg,
                           const BlockParams<T>& params, const FastPath fast) {
  if (fast.const_denom) {
    block<true, true>(cfg, params);
  } else if (fast.nan_free) {
    block<true, false>(cfg, params);
  } else {
    block<false, false>(cfg, params);
  }
}

}  // namespace gower_blocked

// EOF
