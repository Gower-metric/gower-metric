// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/ndarray.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "../utils/enums.hpp"
#include "../utils/half_float.hpp"

using ValueOrders = std::map<int, std::vector<std::string>>;
using OrdinalCounts = std::map<int, std::vector<double>>;

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
  std::string missing_strategy;
  std::string scale_method;
  std::string categorical_ordinal_calculation_type;
  bool conditional_distances = false;
  int conditional_distances_threshold_coeff = 1;
  std::string discretization;
  double silverman_constant = 1.06;
  std::optional<int> k_neighbors;
  std::string handle_unseen_binary_asymmetric;
  std::string handle_unseen_binary_symmetric;
  std::string handle_unseen_categorical_nominal;
  std::string handle_unseen_categorical_ordinal;
  std::string out_of_range;
  bool skip_out_of_range_validation = false;
  ValueOrders categorical_ordinal_values_order;
  ValueOrders binary_asymmetric_value_order;
  ValueOrders binary_symmetric_value_order;
  OrdinalCounts ordinal_counts;
};

template <typename T>
struct OrdinalRankMeta {
  std::vector<T> mid;
  T denom = 0;
  T podani_denom = 0;
  bool use_podani = false;
};

template <typename T>
struct CppConfig {
  std::vector<FeatureTypes> feature_types;
  std::vector<char> is_categorical_feature;
  std::vector<T> feature_weights;
  std::vector<T> ranges;
  std::vector<T> bandwidths;

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
  ScaleMethods scale_method = ScaleMethods::RANGE;
  CategoricalOrdinalCalculationTypes categorical_ordinal_calculation_type =
      CategoricalOrdinalCalculationTypes::KAUFMAN;
  bool conditional_distances = false;
  int conditional_distances_threshold_coeff = 1;
  Discretization discretization = Discretization::NONE;
  double silverman_constant = 1.06;
  std::optional<int> k_neighbors;
  HandleUnseenBinaryAsymmetric handle_unseen_binary_asymmetric =
      HandleUnseenBinaryAsymmetric::ERROR;
  HandleUnseenBinarySymmetric handle_unseen_binary_symmetric =
      HandleUnseenBinarySymmetric::ERROR;
  HandleUnseenCategoricalNominal handle_unseen_categorical_nominal =
      HandleUnseenCategoricalNominal::ERROR;
  HandleUnseenCategoricalOrdinal handle_unseen_categorical_ordinal =
      HandleUnseenCategoricalOrdinal::ERROR;
  OutOfRangeStrategy out_of_range = OutOfRangeStrategy::ERROR;
  bool skip_out_of_range_validation = false;
  ValueOrders categorical_ordinal_values_order;
  ValueOrders binary_asymmetric_value_order;
  ValueOrders binary_symmetric_value_order;
  std::vector<OrdinalRankMeta<T>> ordinal_meta;

  void configure_arguments(const CppConfigData<T>& data);

  auto calculate_distance(const T* x, const T* y) const -> T;
};

// EOF
