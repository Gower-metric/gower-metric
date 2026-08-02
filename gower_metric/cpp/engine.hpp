// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <stdexcept>

#include "components/binary_asymmetric.hpp"
#include "components/binary_symmetric.hpp"
#include "components/categorical_nominal.hpp"
#include "components/categorical_ordinal.hpp"
#include "components/numeric.hpp"
#include "config/gower_config.hpp"
#include "utils/nan.hpp"

namespace gower_detail {

template <typename T>
struct WeightedSum {
  T diff = 0;
  T weight = 0;

  void add(const T w, const T d) {
    diff += w * d;
    weight += w;
  }
};

template <typename T>
inline void accumulate_missing(const MissingStrategy strategy, const T w,
                               WeightedSum<T>& acc) {
  if (strategy == MissingStrategy::IGNORE) return;
  if (strategy == MissingStrategy::RAISE_ERROR) {
    throw std::invalid_argument(
        "Missing values detected in data. Set missing_strategy='ignore' or "
        "'max_dist' to handle them.");
  }
  acc.add(w, T(1));
}

}  // namespace gower_detail

template <typename T>
auto CppConfig<T>::calculate_distance(const T* x, const T* y) const -> T {
  using gower_detail::accumulate_missing;
  using gower_detail::is_missing;
  using gower_detail::WeightedSum;

  WeightedSum<T> cat;
  WeightedSum<T> num;

  const T* weights = feature_weights.data();

  for (const int i : nominal_group) {
    const T w = uniform_weights ? T(1) : weights[i];
    if (is_missing(x[i], y[i])) {
      accumulate_missing(missing_strategy, w, cat);
      continue;
    }
    cat.add(w, compute_categorical_nominal_diff(
                   CategoricalNominalParams<T>{x, y, i}));
  }

  for (const int i : ordinal_group) {
    const T w = uniform_weights ? T(1) : weights[i];
    if (is_missing(x[i], y[i])) {
      accumulate_missing(missing_strategy, w, cat);
      continue;
    }
    const OrdinalRankMeta<T>& meta = ordinal_meta[i];
    cat.add(w, compute_categorical_ordinal_diff(CategoricalOrdinalParams<T>{
                   x, y, i, meta.mid.data(), meta.denom, meta.podani_denom,
                   meta.use_podani}));
  }

  for (const int i : binary_symmetric_group) {
    const T w = uniform_weights ? T(1) : weights[i];
    if (is_missing(x[i], y[i])) {
      accumulate_missing(missing_strategy, w, cat);
      continue;
    }
    cat.add(w,
            compute_binary_symmetric_diff(BinarySymmetricParams<T>{x, y, i}));
  }

  for (const int i : binary_asymmetric_group) {
    const T w = uniform_weights ? T(1) : weights[i];
    if (is_missing(x[i], y[i])) {
      accumulate_missing(missing_strategy, w, cat);
      continue;
    }
    const auto result =
        compute_binary_asymmetric_diff(BinaryAsymmetricParams<T>{x, y, i});
    if (!result.present) continue;
    cat.add(w, result.diff);
  }

  const T nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());

  bool cat_over_threshold = false;
  if (conditional_distances) {
    if (cat.weight <= T(0)) return nan;
    const T threshold = static_cast<T>(conditional_distances_threshold_coeff) /
                        static_cast<T>(p_cat);
    cat_over_threshold = cat.diff / cat.weight > threshold;
    if (cat_over_threshold &&
        missing_strategy != MissingStrategy::RAISE_ERROR) {
      return T(1);
    }
  }

  for (const int i : numeric_group) {
    const T w = uniform_weights ? T(1) : weights[i];
    if (is_missing(x[i], y[i])) {
      accumulate_missing(missing_strategy, w, num);
      continue;
    }
    num.add(w, compute_numeric_diff(NumericParams<T>{x, y, ranges.data(),
                                                     bandwidths.data(), i}));
  }

  if (conditional_distances) {
    if (cat_over_threshold) return T(1);
    return num.weight > T(0) ? num.diff / num.weight : nan;
  }

  const T total_w = cat.weight + num.weight;
  return total_w > T(0) ? (cat.diff + num.diff) / total_w : nan;
}

// EOF
