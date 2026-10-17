// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../engine.hpp"
#include "blocked.hpp"

constexpr double GOWER_MATRIX_PARALLEL_WORK = 1U << 19;

namespace gower_detail {

template <typename T>
inline void raise_if_missing(const CppConfig<T>& cfg, const T* data,
                             const std::size_t count) {
  if (cfg.missing_strategy != MissingStrategy::RAISE_ERROR) return;
  if (!gower_blocked::has_missing(data, count)) return;
  throw std::invalid_argument(
      "Missing values detected in data. Set missing_strategy='ignore' or "
      "'max_dist' to handle them.");
}

template <typename Layout, typename T, typename TOut>
inline void mark_undefined_diagonal(const CppConfig<T>& cfg, const T* data,
                                    TOut* out, const std::size_t n_rows) {
  if constexpr (Layout::kHasDiagonal) {
    const auto nf = static_cast<std::size_t>(cfg.n_features);
    const bool raises = cfg.missing_strategy == MissingStrategy::RAISE_ERROR;
    const auto nan =
        static_cast<TOut>(std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < n_rows; ++i) {
      const T* row = data + (i * nf);
      if (raises && gower_blocked::has_missing(row, nf)) continue;
      if (!gower_detail::is_present(cfg.calculate_distance(row, row))) {
        Layout::set_diagonal(out, n_rows, i, nan);
      }
    }
  }
}

template <typename T>
struct TileView {
  const T* data;
  std::size_t stride;
  std::size_t i0;
  std::size_t rows;
  std::size_t j0;
  std::size_t len;
  bool diagonal;
};

inline auto condensed_index(const std::size_t i, const std::size_t j,
                            const std::size_t n) -> std::size_t {
  return (n * i) - (i * (i + 1) / 2) + (j - i - 1);
}

struct SquareLayout {
  template <typename T, typename TOut>
  static void scatter(const TileView<T>& tile, TOut* out, const std::size_t n) {
    const std::size_t j0 = tile.j0;
    const std::size_t len = tile.len;

    for (std::size_t r = 0; r < tile.rows; ++r) {
      const std::size_t i = tile.i0 + r;
      const std::size_t b_begin =
          tile.diagonal && i >= j0 ? std::min(i + 1 - j0, len) : 0;
      TOut* out_i = out + (i * n) + j0;
      const T* src = tile.data + (r * tile.stride);
      for (std::size_t b = b_begin; b < len; ++b) {
        out_i[b] = static_cast<TOut>(src[b]);
      }
    }

    for (std::size_t b = 0; b < len; ++b) {
      const std::size_t j = j0 + b;
      TOut* out_j = out + (j * n) + tile.i0;
      const T* src = tile.data + b;
      const std::size_t r_end = tile.diagonal && j >= tile.i0
                                    ? std::min(j - tile.i0, tile.rows)
                                    : tile.rows;
      for (std::size_t r = 0; r < r_end; ++r) {
        out_j[r] = static_cast<TOut>(src[r * tile.stride]);
      }
    }
  }

  template <typename T, typename TOut>
  static void store(TOut* out, const std::size_t n, const std::size_t i,
                    const std::size_t j, const T v) {
    const auto cell = static_cast<TOut>(v);
    out[(i * n) + j] = cell;
    out[(j * n) + i] = cell;
  }

  template <typename TOut>
  static void fill_diagonal(TOut* out, const std::size_t n, const TOut v) {
    for (std::size_t i = 0; i < n; ++i) out[(i * n) + i] = v;
  }

  static constexpr bool kHasDiagonal = true;

  template <typename TOut>
  static void set_diagonal(TOut* out, const std::size_t n, const std::size_t i,
                           const TOut v) {
    out[(i * n) + i] = v;
  }
};

struct CondensedLayout {
  template <typename T, typename TOut>
  static void scatter(const TileView<T>& tile, TOut* out, const std::size_t n) {
    const std::size_t j0 = tile.j0;
    const std::size_t len = tile.len;

    for (std::size_t r = 0; r < tile.rows; ++r) {
      const std::size_t i = tile.i0 + r;
      const std::size_t b_begin =
          tile.diagonal && i >= j0 ? std::min(i + 1 - j0, len) : 0;
      if (b_begin >= len) continue;
      TOut* dst = out + condensed_index(i, j0 + b_begin, n);
      const T* src = tile.data + (r * tile.stride) + b_begin;
      for (std::size_t b = 0; b < len - b_begin; ++b) {
        dst[b] = static_cast<TOut>(src[b]);
      }
    }
  }

  template <typename T, typename TOut>
  static void store(TOut* out, const std::size_t n, const std::size_t i,
                    const std::size_t j, const T v) {
    out[condensed_index(i, j, n)] = static_cast<TOut>(v);
  }

  template <typename TOut>
  static void fill_diagonal(TOut* /*out*/, const std::size_t /*n*/, const TOut
                            /*v*/) {
    // A condensed vector holds no diagonal.
  }

  static constexpr bool kHasDiagonal = false;
};

}  // namespace gower_detail

template <typename Layout, typename T, typename TOut = T>
void compute_gower_upper(const CppConfig<T>& cfg, const T* data, TOut* out,
                         const std::size_t n_rows, bool similarity) {
  const auto nf = static_cast<std::size_t>(cfg.n_features);
  Layout::fill_diagonal(out, n_rows, similarity ? TOut(1) : TOut(0));
  gower_detail::mark_undefined_diagonal<Layout>(cfg, data, out, n_rows);

  if (n_rows < 2) return;
  gower_detail::raise_if_missing(cfg, data, n_rows * nf);

  const bool parallel = static_cast<double>(n_rows) *
                            static_cast<double>(n_rows) *
                            static_cast<double>(nf) >=
                        GOWER_MATRIX_PARALLEL_WORK;

  const bool blocked = gower_blocked::supports_blocked(cfg);
  std::vector<T> soa;
  std::vector<unsigned char> clean;
  std::vector<unsigned char> level_flags;
  gower_blocked::FastPath fast;
  T total_weight = 0;
  if (blocked) {
    gower_blocked::transpose(data, n_rows, nf, soa);
    gower_blocked::mark_clean_columns(soa.data(), n_rows, nf, clean);
    gower_blocked::mark_level_columns(cfg, soa.data(), n_rows, nf, level_flags);
    fast.nan_free =
        std::all_of(clean.begin(), clean.end(),
                    [](const unsigned char c) -> bool { return c != 0; });
    fast.const_denom = gower_blocked::constant_denominator(cfg, fast.nan_free);
    total_weight = gower_blocked::weight_total(cfg);
  }
  const T* soa_data = soa.data();
  const unsigned char* clean_cols = clean.data();
  const unsigned char* level_cols = level_flags.data();

  constexpr std::size_t ROWS = gower_blocked::rows_per_tile<T>();
  constexpr std::size_t BLOCK = gower_blocked::BLOCK;
  const auto n_tiles = static_cast<std::ptrdiff_t>((n_rows + ROWS - 1) / ROWS);

#pragma omp parallel for schedule(dynamic) if (parallel) default(none)         \
    shared(data, out, nf, n_rows, n_tiles, cfg, similarity, blocked, soa_data, \
               clean_cols, level_cols, fast, total_weight)
  for (std::ptrdiff_t ti = 0; ti < n_tiles; ++ti) {
    const auto i0 = static_cast<std::size_t>(ti) * ROWS;
    const std::size_t rows = n_rows - i0 < ROWS ? n_rows - i0 : ROWS;

    if (!blocked) {
      for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t i = i0 + r;
        const T* row_i = data + (i * nf);
        for (std::size_t j = i + 1; j < n_rows; ++j) {
          const T d = cfg.calculate_distance(row_i, data + (j * nf));
          Layout::store(out, n_rows, i, j,
                        similarity ? static_cast<T>(T(1) - d) : d);
        }
      }
      continue;
    }

    alignas(64) std::array<T, ROWS * BLOCK> tile{};
    for (std::size_t j0 = i0; j0 < n_rows; j0 += BLOCK) {
      const std::size_t len = n_rows - j0 < BLOCK ? n_rows - j0 : BLOCK;
      const gower_blocked::BlockParams<T> params{
          soa_data, n_rows,     i0,           soa_data,   n_rows,    j0,
          len,      similarity, total_weight, clean_cols, level_cols};

      if (rows == ROWS) {
        gower_blocked::block_dispatch<ROWS>(cfg, params, fast, tile.data());
      } else {
        for (std::size_t r = 0; r < rows; ++r) {
          gower_blocked::BlockParams<T> one = params;
          one.xi = i0 + r;
          gower_blocked::block_dispatch<1>(cfg, one, fast,
                                           tile.data() + (r * BLOCK));
        }
      }

      Layout::scatter(gower_detail::TileView<T>{tile.data(), BLOCK, i0, rows,
                                                j0, len, j0 < i0 + rows},
                      out, n_rows);
    }
  }
}

template <typename T, typename TOut = T>
void compute_gower_matrix(const CppConfig<T>& cfg, const T* data, TOut* out,
                          const std::size_t n_rows, bool similarity) {
  compute_gower_upper<gower_detail::SquareLayout>(cfg, data, out, n_rows,
                                                  similarity);
}

template <typename T, typename TOut = T>
void compute_gower_matrix_condensed(const CppConfig<T>& cfg, const T* data,
                                    TOut* out, const std::size_t n_rows,
                                    bool similarity) {
  compute_gower_upper<gower_detail::CondensedLayout>(cfg, data, out, n_rows,
                                                     similarity);
}

template <typename T, typename TOut = T>
void compute_gower_matrix_xy(const CppConfig<T>& cfg, const T* x_data,
                             const std::size_t n_x, const T* y_data,
                             const std::size_t n_y, TOut* out,
                             bool similarity) {
  const auto nf = static_cast<std::size_t>(cfg.n_features);
  if (n_x == 0 || n_y == 0) return;
  gower_detail::raise_if_missing(cfg, x_data, n_x * nf);
  gower_detail::raise_if_missing(cfg, y_data, n_y * nf);

  const bool parallel = static_cast<double>(n_x) * static_cast<double>(n_y) *
                            static_cast<double>(nf) >=
                        GOWER_MATRIX_PARALLEL_WORK;

  const bool blocked = gower_blocked::supports_blocked(cfg);
  std::vector<T> x_soa;
  std::vector<T> y_soa;
  std::vector<unsigned char> clean;
  std::vector<unsigned char> level_flags;
  gower_blocked::FastPath fast;
  T total_weight = 0;
  if (blocked) {
    gower_blocked::transpose(x_data, n_x, nf, x_soa);
    gower_blocked::transpose(y_data, n_y, nf, y_soa);
    gower_blocked::mark_clean_columns(x_soa.data(), n_x, nf, clean);
    gower_blocked::intersect_clean_columns(y_soa.data(), n_y, nf, clean);
    gower_blocked::mark_level_columns(cfg, x_soa.data(), n_x, nf, level_flags);
    gower_blocked::intersect_level_columns(cfg, y_soa.data(), n_y, level_flags);
    fast.nan_free =
        std::all_of(clean.begin(), clean.end(),
                    [](const unsigned char c) -> bool { return c != 0; });
    fast.const_denom = gower_blocked::constant_denominator(cfg, fast.nan_free);
    total_weight = gower_blocked::weight_total(cfg);
  }
  const T* x_soa_data = x_soa.data();
  const T* y_soa_data = y_soa.data();
  const unsigned char* clean_cols = clean.data();
  const unsigned char* level_cols = level_flags.data();

  constexpr std::size_t ROWS = gower_blocked::rows_per_tile<T>();
  constexpr std::size_t BLOCK = gower_blocked::BLOCK;
  const auto n_tiles = static_cast<std::ptrdiff_t>((n_x + ROWS - 1) / ROWS);

#pragma omp parallel for schedule(dynamic) if (parallel) default(none)        \
    shared(x_data, y_data, out, nf, n_x, n_y, n_tiles, cfg, similarity,       \
               blocked, x_soa_data, y_soa_data, clean_cols, level_cols, fast, \
               total_weight)
  for (std::ptrdiff_t ti = 0; ti < n_tiles; ++ti) {
    const auto i0 = static_cast<std::size_t>(ti) * ROWS;
    const std::size_t rows = n_x - i0 < ROWS ? n_x - i0 : ROWS;

    if (!blocked) {
      for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t i = i0 + r;
        const T* row_i = x_data + (i * nf);
        TOut* out_i = out + (i * n_y);
        for (std::size_t j = 0; j < n_y; ++j) {
          const T d = cfg.calculate_distance(row_i, y_data + (j * nf));
          out_i[j] =
              static_cast<TOut>(similarity ? static_cast<T>(T(1) - d) : d);
        }
      }
      continue;
    }

    alignas(64) std::array<T, ROWS * BLOCK> tile{};
    for (std::size_t j0 = 0; j0 < n_y; j0 += BLOCK) {
      const std::size_t len = n_y - j0 < BLOCK ? n_y - j0 : BLOCK;
      const gower_blocked::BlockParams<T> params{
          x_soa_data, n_x,        i0,           y_soa_data, n_y,       j0,
          len,        similarity, total_weight, clean_cols, level_cols};

      if (rows == ROWS) {
        gower_blocked::block_dispatch<ROWS>(cfg, params, fast, tile.data());
      } else {
        for (std::size_t r = 0; r < rows; ++r) {
          gower_blocked::BlockParams<T> one = params;
          one.xi = i0 + r;
          gower_blocked::block_dispatch<1>(cfg, one, fast,
                                           tile.data() + (r * BLOCK));
        }
      }

      for (std::size_t r = 0; r < rows; ++r) {
        TOut* out_i = out + ((i0 + r) * n_y) + j0;
        const T* src = tile.data() + (r * BLOCK);
        for (std::size_t b = 0; b < len; ++b) {
          out_i[b] = static_cast<TOut>(src[b]);
        }
      }
    }
  }
}

// EOF
