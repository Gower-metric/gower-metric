// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../engine.hpp"
#include "blocked.hpp"

constexpr double GOWER_MATRIX_PARALLEL_WORK = 1U << 19;
constexpr std::size_t GOWER_MATRIX_TILE = 128;

template <typename T>
void compute_gower_matrix(const CppConfig<T>& cfg, const T* data, T* out,
                          const std::size_t n_rows, bool similarity) {
  const auto nf = static_cast<std::size_t>(cfg.n_features);
  const T diag = similarity ? T(1) : T(0);

  for (std::size_t i = 0; i < n_rows; ++i) {
    out[i * n_rows + i] = diag;
  }

  const bool parallel = static_cast<double>(n_rows) *
                            static_cast<double>(n_rows) *
                            static_cast<double>(nf) >=
                        GOWER_MATRIX_PARALLEL_WORK;

  const bool blocked = gower_blocked::supports_blocked(cfg);
  std::vector<T> soa;
  gower_blocked::FastPath fast;
  T total_weight = 0;
  if (blocked) {
    gower_blocked::transpose(data, n_rows, nf, soa);
    fast.nan_free = !gower_blocked::has_missing(data, n_rows * nf);
    fast.const_denom = fast.nan_free && cfg.binary_asymmetric_group.empty();
    total_weight = gower_blocked::weight_total(cfg);
  }

  if (n_rows >= 2) {
    const std::size_t last = n_rows - 1;
    const T* soa_data = soa.data();
#pragma omp parallel for schedule(dynamic) if (parallel) default(none)      \
    shared(data, out, nf, n_rows, last, cfg, similarity, blocked, soa_data, \
               fast, total_weight)
    for (std::size_t i = 0; i < last; ++i) {
      T* out_i = out + i * n_rows;
      if (blocked) {
        for (std::size_t j = i + 1; j < n_rows; j += gower_blocked::BLOCK) {
          const std::size_t len = std::min(gower_blocked::BLOCK, n_rows - j);
          gower_blocked::block_dispatch(
              cfg,
              gower_blocked::BlockParams<T>{soa_data, n_rows, i, soa_data,
                                            n_rows, j, len, out_i + j,
                                            similarity, total_weight},
              fast);
        }
      } else {
        const T* row_i = data + i * nf;
        for (std::size_t j = i + 1; j < n_rows; ++j) {
          const T d = cfg.calculate_distance(row_i, data + j * nf);
          out_i[j] = similarity ? static_cast<T>(T(1) - d) : d;
        }
      }
    }
  }

#pragma omp parallel for schedule(dynamic) if (parallel) default(none) \
    shared(out, n_rows)
  for (std::size_t i = 0; i < n_rows; i += GOWER_MATRIX_TILE) {
    const std::size_t i_max = std::min(i + GOWER_MATRIX_TILE, n_rows);

    for (std::size_t j = 0; j < i; j += GOWER_MATRIX_TILE) {
      const std::size_t j_max = std::min(j + GOWER_MATRIX_TILE, n_rows);
      for (std::size_t row = i; row < i_max; ++row) {
        T* out_row = out + row * n_rows;
        for (std::size_t col = j; col < j_max; ++col) {
          out_row[col] = out[col * n_rows + row];
        }
      }
    }

    for (std::size_t row = i; row < i_max; ++row) {
      T* out_row = out + row * n_rows;
      for (std::size_t col = i; col < row; ++col) {
        out_row[col] = out[col * n_rows + row];
      }
    }
  }
}

template <typename T>
void compute_gower_matrix_xy(const CppConfig<T>& cfg, const T* x_data,
                             const std::size_t n_x, const T* y_data,
                             const std::size_t n_y, T* out, bool similarity) {
  const auto nf = static_cast<std::size_t>(cfg.n_features);

  const bool parallel = static_cast<double>(n_x) * static_cast<double>(n_y) *
                            static_cast<double>(nf) >=
                        GOWER_MATRIX_PARALLEL_WORK;

  const bool blocked = gower_blocked::supports_blocked(cfg);
  std::vector<T> x_soa;
  std::vector<T> y_soa;
  gower_blocked::FastPath fast;
  T total_weight = 0;
  if (blocked) {
    gower_blocked::transpose(x_data, n_x, nf, x_soa);
    gower_blocked::transpose(y_data, n_y, nf, y_soa);
    fast.nan_free = !gower_blocked::has_missing(x_data, n_x * nf) &&
                    !gower_blocked::has_missing(y_data, n_y * nf);
    fast.const_denom = fast.nan_free && cfg.binary_asymmetric_group.empty();
    total_weight = gower_blocked::weight_total(cfg);
  }
  const T* x_soa_data = x_soa.data();
  const T* y_soa_data = y_soa.data();

#pragma omp parallel for schedule(dynamic) if (parallel) default(none)  \
    shared(x_data, y_data, out, nf, n_x, n_y, cfg, similarity, blocked, \
               x_soa_data, y_soa_data, fast, total_weight)
  for (std::size_t i = 0; i < n_x; ++i) {
    T* out_i = out + i * n_y;
    if (blocked) {
      for (std::size_t j = 0; j < n_y; j += gower_blocked::BLOCK) {
        const std::size_t len = std::min(gower_blocked::BLOCK, n_y - j);
        gower_blocked::block_dispatch(
            cfg,
            gower_blocked::BlockParams<T>{x_soa_data, n_x, i, y_soa_data, n_y,
                                          j, len, out_i + j, similarity,
                                          total_weight},
            fast);
      }
    } else {
      const T* row_i = x_data + i * nf;
      for (std::size_t j = 0; j < n_y; ++j) {
        const T d = cfg.calculate_distance(row_i, y_data + j * nf);
        out_i[j] = similarity ? static_cast<T>(T(1) - d) : d;
      }
    }
  }
}

// EOF
