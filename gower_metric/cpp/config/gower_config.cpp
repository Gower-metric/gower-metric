// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#include "gower_config.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "../utils/config_converter.hpp"

namespace {

template <typename T>
auto view_size(const ConfigVector<T>& v) -> std::size_t {
  return v.is_valid() ? static_cast<std::size_t>(v.shape(0)) : 0;
}

}  // namespace

template <typename T>
void CppConfig<T>::configure_arguments(const CppConfigData<T>& data) {
  const std::lock_guard<std::mutex> lock(configuring);
  const auto n = data.feature_types.size();

  const auto weights_size = view_size(data.feature_weights);
  const auto ranges_size = view_size(data.ranges);
  const auto bandwidths_size = view_size(data.bandwidths);

  if (weights_size != 0 && weights_size != n) {
    throw std::invalid_argument(
        "feature_weights length (" + std::to_string(weights_size) +
        ") does not match feature_types length (" + std::to_string(n) + ")");
  }
  if (ranges_size != n) {
    throw std::invalid_argument(
        "ranges length (" + std::to_string(ranges_size) +
        ") does not match feature_types length (" + std::to_string(n) + ")");
  }
  if (bandwidths_size != n) {
    throw std::invalid_argument(
        "bandwidths length (" + std::to_string(bandwidths_size) +
        ") does not match feature_types length (" + std::to_string(n) + ")");
  }
  if (view_size(data.lower) != n || view_size(data.upper) != n) {
    throw std::invalid_argument(
        "lower and upper must have feature_types length");
  }
  if (data.conditional_distances_threshold_coeff < 1) {
    throw std::invalid_argument(
        "conditional_distances_threshold_coeff must be at least 1, got " +
        std::to_string(data.conditional_distances_threshold_coeff));
  }

  n_features = static_cast<int>(n);

  feature_types.clear();
  feature_types.reserve(n);
  for (const auto& t : data.feature_types) {
    feature_types.push_back(string_to_feature_type(t));
  }

  p_cat = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (in_conditional_first_step(feature_types[i])) ++p_cat;
  }

  numeric_group.clear();
  nominal_group.clear();
  ordinal_group.clear();
  binary_symmetric_group.clear();
  binary_asymmetric_group.clear();
  for (std::size_t i = 0; i < n; ++i) {
    const auto idx = static_cast<int>(i);
    switch (feature_types[i]) {
      case FeatureTypes::NUMERIC:
      case FeatureTypes::RATIO_SCALE_INTERVAL:
        numeric_group.push_back(idx);
        break;
      case FeatureTypes::CATEGORICAL_NOMINAL:
        nominal_group.push_back(idx);
        break;
      case FeatureTypes::CATEGORICAL_ORDINAL:
        ordinal_group.push_back(idx);
        break;
      case FeatureTypes::BINARY_SYMMETRIC:
        binary_symmetric_group.push_back(idx);
        break;
      case FeatureTypes::BINARY_ASYMMETRIC:
        binary_asymmetric_group.push_back(idx);
        break;
      default:
        throw std::invalid_argument("Unknown feature type at index " +
                                    std::to_string(idx));
    }
  }

  if (weights_size == 0) {
    feature_weights.clear();
  } else {
    const T* w_ptr = data.feature_weights.data();
    feature_weights.assign(w_ptr, w_ptr + weights_size);
  }
  uniform_weights = true;
  for (const T w : feature_weights) {
    if (w != T(1)) {
      uniform_weights = false;
      break;
    }
  }

  const T* r_ptr = data.ranges.data();
  ranges.assign(r_ptr, r_ptr + ranges_size);
  const T* b_ptr = data.bandwidths.data();
  bandwidths.assign(b_ptr, b_ptr + bandwidths_size);
  lower.assign(data.lower.data(), data.lower.data() + n);
  upper.assign(data.upper.data(), data.upper.data() + n);
  out_of_range = string_to_out_of_range_strategy(data.out_of_range);
  missing_strategy = string_to_missing_strategy(data.missing_strategy);
  categorical_ordinal_calculation_type =
      string_to_categorical_ordinal_calculation_type(
          data.categorical_ordinal_calculation_type);
  conditional_distances = data.conditional_distances;
  conditional_distances_threshold_coeff =
      data.conditional_distances_threshold_coeff;
  handle_unseen_binary_asymmetric =
      string_to_handle_unseen<HandleUnseenBinaryAsymmetric>(
          data.handle_unseen_binary_asymmetric);
  handle_unseen_binary_symmetric =
      string_to_handle_unseen<HandleUnseenBinarySymmetric>(
          data.handle_unseen_binary_symmetric);
  handle_unseen_categorical_nominal =
      string_to_handle_unseen<HandleUnseenCategoricalNominal>(
          data.handle_unseen_categorical_nominal);
  handle_unseen_categorical_ordinal =
      string_to_handle_unseen<HandleUnseenCategoricalOrdinal>(
          data.handle_unseen_categorical_ordinal);

  ordinal_meta.assign(n, OrdinalRankMeta<T>{});
  for (const auto& [j, counts] : data.ordinal_counts) {
    if (j < 0 || static_cast<std::size_t>(j) >= n) {
      throw std::invalid_argument("ordinal_counts feature index " +
                                  std::to_string(j) + " out of range [0, " +
                                  std::to_string(n) + ")");
    }
    if (counts.empty()) continue;

    OrdinalRankMeta<T>& meta = ordinal_meta[j];
    const std::size_t c = counts.size();
    meta.levels = c;
    meta.cum.assign(c + 1, 0.0);
    for (std::size_t r = 0; r < c; ++r) {
      meta.cum[r + 1] = meta.cum[r] + counts[r];
    }

    std::size_t lo = 0;
    while (lo < c && counts[lo] <= 0.0) ++lo;
    std::size_t hi = c - 1;
    while (hi > lo && counts[hi] <= 0.0) --hi;
    meta.varies = lo < c && hi > lo;
    if (!meta.varies) continue;

    meta.lo = lo;
    meta.hi = hi;
    meta.denom = static_cast<T>(hi - lo);
    meta.podani_denom = 1.0 + (meta.cum[hi] - meta.cum[lo + 1]);
  }

  domains.assign(n, FeatureDomain<T>{});
  checked_features.clear();
  for (std::size_t f = 0; f < n; ++f) {
    FeatureDomain<T>& domain = domains[f];
    const FeatureTypes type = feature_types[f];
    const auto fitted = data.category_values.find(static_cast<int>(f));
    if (type == FeatureTypes::CATEGORICAL_ORDINAL) {
      if (ordinal_meta[f].levels == 0) continue;
      for (std::size_t l = 0; l < ordinal_meta[f].levels; ++l) {
        domain.values.push_back(static_cast<T>(l));
      }
      domain.strategy = as_unseen_strategy(handle_unseen_categorical_ordinal);
    } else if (fitted == data.category_values.end()) {
      continue;
    } else {
      for (const double v : fitted->second) {
        domain.values.push_back(static_cast<T>(v));
      }
      domain.strategy =
          type == FeatureTypes::CATEGORICAL_NOMINAL
              ? as_unseen_strategy(handle_unseen_categorical_nominal)
          : type == FeatureTypes::BINARY_SYMMETRIC
              ? as_unseen_strategy(handle_unseen_binary_symmetric)
              : as_unseen_strategy(handle_unseen_binary_asymmetric);
      if (type != FeatureTypes::CATEGORICAL_NOMINAL) {
        domain.room = 2 - domain.values.size();
        if (domain.room == 0) domain.positive = domain.values[1];
      }
      std::sort(domain.values.begin(), domain.values.end());
    }
    for (std::size_t i = 0; i < domain.values.size(); ++i) {
      domain.positions &= domain.values[i] == static_cast<T>(i);
    }
    checked_features.push_back(static_cast<int>(f));
  }

  std::vector<char> excluded(n, 0);
  for (const int f : data.excluded_features) {
    if (f < 0 || static_cast<std::size_t>(f) >= n) {
      throw std::invalid_argument("excluded_features index " +
                                  std::to_string(f) + " out of range [0, " +
                                  std::to_string(n) + ")");
    }
    excluded[static_cast<std::size_t>(f)] = 1;
  }
  const auto drop = [&excluded](std::vector<int>& group) -> void {
    group.erase(std::remove_if(group.begin(), group.end(),
                               [&excluded](const int f) -> bool {
                                 return excluded[static_cast<std::size_t>(f)] !=
                                        0;
                               }),
                group.end());
  };
  drop(numeric_group);
  drop(nominal_group);
  drop(ordinal_group);
  drop(binary_symmetric_group);
  drop(binary_asymmetric_group);
  ordinal_group.erase(std::remove_if(ordinal_group.begin(), ordinal_group.end(),
                                     [this](const int f) -> bool {
                                       return !ordinal_meta[f].varies;
                                     }),
                      ordinal_group.end());
}

template struct CppConfig<double>;
template struct CppConfig<float>;
#ifdef GOWER_HAS_FLOAT16
template struct CppConfig<float16>;
#endif
