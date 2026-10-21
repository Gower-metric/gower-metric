// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "../config/gower_config.hpp"
#include "../utils/arith.hpp"
#include "../utils/nan.hpp"

namespace gower_blocked {

constexpr std::size_t BLOCK = 128;

template <typename T>
constexpr auto rows_per_tile() -> std::size_t {
  constexpr std::size_t by_line = 64 / sizeof(T);
  return by_line < 4 ? 4 : by_line;
}

template <typename T>
auto supports_blocked(const CppConfig<T>& cfg) -> bool {
  if (cfg.conditional_distances) return false;
  if (cfg.missing_strategy == MissingStrategy::RAISE_ERROR) return false;
  return true;
}

template <typename T>
inline auto has_missing(const T* data, const std::size_t count) -> bool {
  for (std::size_t i = 0; i < count; ++i) {
    if (!gower_detail::is_present(data[i])) return true;
  }
  return false;
}

template <typename T>
inline void mark_clean_columns(const T* soa, const std::size_t n,
                               const std::size_t nf,
                               std::vector<unsigned char>& clean) {
  clean.assign(nf, 1);
  for (std::size_t f = 0; f < nf; ++f) {
    clean[f] = has_missing(soa + (f * n), n) ? 0 : 1;
  }
}

template <typename T>
inline void intersect_clean_columns(const T* soa, const std::size_t n,
                                    const std::size_t nf,
                                    std::vector<unsigned char>& clean) {
  for (std::size_t f = 0; f < nf; ++f) {
    if (clean[f] == 0) continue;
    clean[f] = has_missing(soa + (f * n), n) ? 0 : 1;
  }
}

template <typename T>
inline auto only_declared_levels(const T* data, const std::size_t count,
                                 const std::size_t levels) -> bool {
  for (std::size_t i = 0; i < count; ++i) {
    const T v = data[i];
    if (gower_detail::is_present(v) &&
        gower_detail::level_index(v, levels) == levels) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline void intersect_level_columns(const CppConfig<T>& cfg, const T* soa,
                                    const std::size_t n,
                                    std::vector<unsigned char>& flags) {
  if (cfg.categorical_ordinal_calculation_type !=
      CategoricalOrdinalCalculationTypes::PODANI) {
    return;
  }
  for (const int f : cfg.ordinal_group) {
    const auto col = static_cast<std::size_t>(f);
    if (flags[col] == 0) continue;
    flags[col] =
        only_declared_levels(soa + (col * n), n, cfg.ordinal_meta[col].levels)
            ? 1
            : 0;
  }
}

template <typename T>
inline void mark_level_columns(const CppConfig<T>& cfg, const T* soa,
                               const std::size_t n, const std::size_t nf,
                               std::vector<unsigned char>& flags) {
  flags.assign(nf, 1);
  intersect_level_columns(cfg, soa, n, flags);
}

template <typename Body>
inline void with_clean(const bool clean, Body&& body) {
  if (clean) {
    body(std::true_type{});
  } else {
    body(std::false_type{});
  }
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
  bool similarity;
  T total_weight;
  const unsigned char* clean_cols;
  const unsigned char* level_cols;
};

template <bool NanFree, bool ConstDenom, std::size_t ROWS, typename T>
inline void block(const CppConfig<T>& cfg, const BlockParams<T>& params,
                  T* __restrict tile) {
  const T* x_soa = params.x_soa;
  const std::size_t x_stride = params.x_stride;
  const std::size_t xi = params.xi;
  const T* y_soa = params.y_soa;
  const std::size_t y_stride = params.y_stride;
  const std::size_t j0 = params.j0;
  const std::size_t len = params.len;
  const bool similarity = params.similarity;

  const bool uniform = cfg.uniform_weights;
  const bool max_dist = cfg.missing_strategy == MissingStrategy::MAX_DISTANCE;
  const T* weights = cfg.feature_weights.data();

  alignas(64) std::array<T, ROWS * BLOCK> cat_diff{};
  alignas(64) std::array<T, ROWS * BLOCK> num_diff{};
  alignas(64) std::array<T, ConstDenom ? 1 : ROWS * BLOCK> cat_w{};
  alignas(64) std::array<T, ConstDenom ? 1 : ROWS * BLOCK> num_w{};

  const auto x_at = [&](const int f, const std::size_t r) -> T {
    return x_soa[static_cast<std::size_t>(f) * x_stride + xi + r];
  };
  const auto y_col = [&](const int f) -> const T* {
    return y_soa + static_cast<std::size_t>(f) * y_stride + j0;
  };
  const auto weight_of = [&](const int f) -> T {
    return uniform ? T(1) : weights[f];
  };

  const auto clean_col = [&](const int f) -> bool {
    return NanFree || params.clean_cols[f] != 0;
  };

  for (const int f : cfg.nominal_group) {
    const T* __restrict col = y_col(f);
    const T wf = weight_of(f);
    with_clean(clean_col(f), [&](auto tag) -> void {
      constexpr bool Clean = decltype(tag)::value;
      for (std::size_t r = 0; r < ROWS; ++r) {
        const T xv = x_at(f, r);
        const bool x_ok = Clean || gower_detail::is_present(xv);
        T* __restrict acc = cat_diff.data() + r * BLOCK;
        [[maybe_unused]] T* __restrict den =
            cat_w.data() + (ConstDenom ? 0 : r * BLOCK);
        for (std::size_t b = 0; b < len; ++b) {
          const T yv = col[b];
          const bool ok =
              NanFree || Clean || (x_ok && gower_detail::is_present(yv));
          const T w = (ok || max_dist) ? wf : T(0);
          acc[b] += w * (ok ? (xv == yv ? T(0) : T(1)) : T(1));
          if constexpr (!ConstDenom) den[b] += w;
        }
      }
    });
  }

  for (const int f : cfg.ordinal_group) {
    const T* __restrict col = y_col(f);
    const T wf = weight_of(f);
    const OrdinalRankMeta<T>& ord = cfg.ordinal_meta[f];
    const T denom = ord.denom;
    const double* __restrict cum = ord.cum.data();
    const std::size_t levels = ord.levels;
    const double podani_denom = ord.podani_denom;
    const auto mode = cfg.categorical_ordinal_calculation_type;
    with_clean(clean_col(f), [&](auto tag) -> void {
      constexpr bool Clean = decltype(tag)::value;
      for (std::size_t r = 0; r < ROWS; ++r) {
        const T xv = x_at(f, r);
        const bool x_ok = Clean || gower_detail::is_present(xv);
        T* __restrict acc = cat_diff.data() + r * BLOCK;
        [[maybe_unused]] T* __restrict den =
            cat_w.data() + (ConstDenom ? 0 : r * BLOCK);
        if (mode == CategoricalOrdinalCalculationTypes::PODANI) {
          with_clean(params.level_cols[f] != 0, [&](auto level_tag) -> void {
            constexpr bool Positions = decltype(level_tag)::value;
            const std::size_t lx =
                Positions ? (x_ok ? static_cast<std::size_t>(xv) : 0)
                          : gower_detail::level_index(xv, levels);
            const bool x_in = Positions || lx != levels;
            const std::size_t ax = x_in ? lx : 0;
            for (std::size_t b = 0; b < len; ++b) {
              const T yv = col[b];
              const bool ok =
                  NanFree || Clean || (x_ok && gower_detail::is_present(yv));
              const T w = (ok || max_dist) ? wf : T(0);
              const std::size_t ly =
                  Positions ? (ok ? static_cast<std::size_t>(yv) : 0)
                            : gower_detail::level_index(yv, levels);
              const bool y_in = Positions || ly != levels;
              const std::size_t ay = y_in ? ly : 0;
              const bool x_lower = xv < yv;
              const std::size_t a = x_lower ? ax : ay;
              const std::size_t hi_idx = x_lower ? ay : ax;
              const double raw =
                  (1.0 + (cum[hi_idx] - cum[a + 1])) / podani_denom;
              const T clipped =
                  (x_in && y_in) ? static_cast<T>(raw < 1.0 ? raw : 1.0) : T(1);
              const T d = xv == yv ? T(0) : clipped;
              acc[b] += w * (ok ? d : T(1));
              if constexpr (!ConstDenom) den[b] += w;
            }
          });
        } else {
          for (std::size_t b = 0; b < len; ++b) {
            const T yv = col[b];
            const bool ok =
                NanFree || Clean || (x_ok && gower_detail::is_present(yv));
            const T w = (ok || max_dist) ? wf : T(0);
            const T d = static_cast<T>(gower_detail::abs_diff(xv, yv) / denom);
            const T clipped = d < T(1) ? d : T(1);
            acc[b] += w * (ok ? clipped : T(1));
            if constexpr (!ConstDenom) den[b] += w;
          }
        }
      }
    });
  }

  for (const int f : cfg.binary_symmetric_group) {
    const T* __restrict col = y_col(f);
    const T wf = weight_of(f);
    const T positive = cfg.domains[f].positive;
    with_clean(clean_col(f), [&](auto tag) -> void {
      constexpr bool Clean = decltype(tag)::value;
      for (std::size_t r = 0; r < ROWS; ++r) {
        const T xv = x_at(f, r);
        const bool x_ok = Clean || gower_detail::is_present(xv);
        const bool x_pos = xv == positive;
        T* __restrict acc = cat_diff.data() + r * BLOCK;
        [[maybe_unused]] T* __restrict den =
            cat_w.data() + (ConstDenom ? 0 : r * BLOCK);
        for (std::size_t b = 0; b < len; ++b) {
          const T yv = col[b];
          const bool ok =
              NanFree || Clean || (x_ok && gower_detail::is_present(yv));
          const T w = (ok || max_dist) ? wf : T(0);
          acc[b] += w * (ok ? (x_pos == (yv == positive) ? T(0) : T(1)) : T(1));
          if constexpr (!ConstDenom) den[b] += w;
        }
      }
    });
  }

  for (const int f : cfg.binary_asymmetric_group) {
    const T* __restrict col = y_col(f);
    const T wf = weight_of(f);
    const T positive = cfg.domains[f].positive;
    with_clean(clean_col(f), [&](auto tag) -> void {
      constexpr bool Clean = decltype(tag)::value;
      for (std::size_t r = 0; r < ROWS; ++r) {
        const T xv = x_at(f, r);
        const bool x_ok = Clean || gower_detail::is_present(xv);
        const bool x_pos = xv == positive;
        T* __restrict acc = cat_diff.data() + r * BLOCK;
        [[maybe_unused]] T* __restrict den =
            cat_w.data() + (ConstDenom ? 0 : r * BLOCK);
        for (std::size_t b = 0; b < len; ++b) {
          const T yv = col[b];
          const bool ok =
              NanFree || Clean || (x_ok && gower_detail::is_present(yv));
          const bool y_pos = yv == positive;
          const bool counted = ok ? (x_pos || y_pos) : max_dist;
          const T w = counted ? wf : T(0);
          acc[b] += w * (ok ? (x_pos != y_pos ? T(1) : T(0)) : T(1));
          if constexpr (!ConstDenom) den[b] += w;
        }
      }
    });
  }

  for (const int f : cfg.numeric_group) {
    const T* __restrict col = y_col(f);
    const T wf = weight_of(f);
    const T range = cfg.ranges[f];
    const bool scaled = range > T(0);
    const T band = cfg.bandwidths[f];
    const bool banded = band > T(0);
    with_clean(clean_col(f), [&](auto tag) -> void {
      constexpr bool Clean = decltype(tag)::value;
      for (std::size_t r = 0; r < ROWS; ++r) {
        const T xv = x_at(f, r);
        const bool x_ok = Clean || gower_detail::is_present(xv);
        T* __restrict acc = num_diff.data() + r * BLOCK;
        [[maybe_unused]] T* __restrict den =
            num_w.data() + (ConstDenom ? 0 : r * BLOCK);
        if (scaled && !banded) {
          for (std::size_t b = 0; b < len; ++b) {
            const T yv = col[b];
            const bool ok =
                NanFree || Clean || (x_ok && gower_detail::is_present(yv));
            const T w = (ok || max_dist) ? wf : T(0);
            const T d = std::min<T>(
                static_cast<T>(gower_detail::abs_diff(xv, yv) / range), T(1));
            acc[b] += w * (ok ? d : T(1));
            if constexpr (!ConstDenom) den[b] += w;
          }
        } else if (scaled) {
          for (std::size_t b = 0; b < len; ++b) {
            const T yv = col[b];
            const bool ok =
                NanFree || Clean || (x_ok && gower_detail::is_present(yv));
            const T w = (ok || max_dist) ? wf : T(0);
            const T raw = gower_detail::abs_diff(xv, yv);
            const T scaled = std::min<T>(static_cast<T>(raw / range), T(1));
            const T d = raw <= band ? T(0) : scaled;
            acc[b] += w * (ok ? d : T(1));
            if constexpr (!ConstDenom) den[b] += w;
          }
        } else {
          for (std::size_t b = 0; b < len; ++b) {
            const T yv = col[b];
            const bool ok =
                NanFree || Clean || (x_ok && gower_detail::is_present(yv));
            const T w = (ok || max_dist) ? wf : T(0);
            const T d = gower_detail::abs_diff(xv, yv) > T(0) ? T(1) : T(0);
            acc[b] += w * (ok ? d : T(1));
            if constexpr (!ConstDenom) den[b] += w;
          }
        }
      }
    });
  }

  const T nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());

  for (std::size_t r = 0; r < ROWS; ++r) {
    const T* __restrict cat = cat_diff.data() + r * BLOCK;
    const T* __restrict num = num_diff.data() + r * BLOCK;
    T* __restrict dst = tile + r * BLOCK;
    if constexpr (ConstDenom) {
      const T total_w = params.total_weight;
      for (std::size_t b = 0; b < len; ++b) {
        const T d = total_w > T(0) ? (cat[b] + num[b]) / total_w : nan;
        dst[b] = similarity ? static_cast<T>(T(1) - d) : d;
      }
    } else {
      const T* __restrict cw = cat_w.data() + r * BLOCK;
      const T* __restrict nw = num_w.data() + r * BLOCK;
      for (std::size_t b = 0; b < len; ++b) {
        const T total_w = cw[b] + nw[b];
        const T q = (cat[b] + num[b]) / total_w;
        const T d = total_w > T(0) ? q : nan;
        dst[b] = similarity ? static_cast<T>(T(1) - d) : d;
      }
    }
  }
}

struct FastPath {
  bool nan_free = false;
  bool const_denom = false;
};

template <typename T>
inline auto constant_denominator(const CppConfig<T>& cfg, const bool nan_free)
    -> bool {
  if (!cfg.binary_asymmetric_group.empty()) return false;
  return nan_free || cfg.missing_strategy == MissingStrategy::MAX_DISTANCE;
}

template <std::size_t ROWS, typename T>
inline void block_dispatch(const CppConfig<T>& cfg,
                           const BlockParams<T>& params, const FastPath fast,
                           T* tile) {
  if (fast.const_denom) {
    if (fast.nan_free) {
      block<true, true, ROWS>(cfg, params, tile);
    } else {
      block<false, true, ROWS>(cfg, params, tile);
    }
  } else if (fast.nan_free) {
    block<true, false, ROWS>(cfg, params, tile);
  } else {
    block<false, false, ROWS>(cfg, params, tile);
  }
}

}  // namespace gower_blocked

// EOF
