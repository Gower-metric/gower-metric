// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/ndarray.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../utils/enums.hpp"
#include "../utils/half_float.hpp"

using OrdinalCounts = std::map<int, std::vector<double>>;
using CategoryValues = std::map<int, std::vector<double>>;

template <typename T>
using ConfigVector =
    nanobind::ndarray<const T, nanobind::ndim<1>, nanobind::c_contig,
                      nanobind::device::cpu>;

template <typename T>
struct CppConfigData {
  std::vector<std::string> feature_types;
  ConfigVector<T> feature_weights;
  ConfigVector<T> ranges;
  ConfigVector<T> bandwidths;

  // out-of-range bounds
  ConfigVector<T> lower;
  ConfigVector<T> upper;

  std::string out_of_range;
  std::string missing_strategy;
  std::string categorical_ordinal_calculation_type;
  bool conditional_distances = false;
  int conditional_distances_threshold_coeff = 1;
  std::string handle_unseen_binary_asymmetric;
  std::string handle_unseen_binary_symmetric;
  std::string handle_unseen_categorical_nominal;
  std::string handle_unseen_categorical_ordinal;
  OrdinalCounts ordinal_counts;
  std::vector<int> excluded_features;
  CategoryValues category_values;
};

template <typename T>
struct OrdinalRankMeta {
  std::vector<double> cum;
  std::size_t levels = 0;
  bool varies = false;
  T denom = 0;
  double podani_denom = 1;
};

template <typename T>
struct FeatureDomain {
  std::vector<T> values;
  bool positions = true;
  UnseenStrategy strategy = UnseenStrategy::ERROR;
  T positive = 1;
  std::size_t room = -1;
};

template <typename T>
struct CppConfig {
  std::vector<FeatureTypes> feature_types;
  std::vector<T> feature_weights;
  std::vector<T> ranges;
  std::vector<T> bandwidths;
  std::vector<T> lower;
  std::vector<T> upper;

  // grouped features based on apperance, better for speed
  std::vector<int> numeric_group;
  std::vector<int> nominal_group;
  std::vector<int> ordinal_group;
  std::vector<int> binary_symmetric_group;
  std::vector<int> binary_asymmetric_group;

  int n_features = 0;
  int p_cat = 0;
  bool uniform_weights = false;
  MissingStrategy missing_strategy = MissingStrategy::IGNORE;
  OutOfRangeStrategy out_of_range = OutOfRangeStrategy::CLIP;
  CategoricalOrdinalCalculationTypes categorical_ordinal_calculation_type =
      CategoricalOrdinalCalculationTypes::KAUFMAN;
  bool conditional_distances = false;
  int conditional_distances_threshold_coeff = 1;
  UnseenStrategy handle_unseen_binary_asymmetric = UnseenStrategy::ERROR;
  UnseenStrategy handle_unseen_binary_symmetric = UnseenStrategy::ERROR;
  UnseenStrategy handle_unseen_categorical_nominal = UnseenStrategy::ERROR;
  UnseenStrategy handle_unseen_categorical_ordinal = UnseenStrategy::ERROR;
  std::vector<OrdinalRankMeta<T>> ordinal_meta;
  std::vector<FeatureDomain<T>> domains;
  std::vector<int> checked_features;
  std::mutex configuring;

  void configure_arguments(const CppConfigData<T>& data);

  auto calculate_distance(const T* x, const T* y) const -> T;
};

// EOF
