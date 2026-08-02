// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

#include "enums.hpp"

inline auto string_to_feature_type(const std::string& str) -> FeatureTypes {
  if (str == "numeric") return FeatureTypes::NUMERIC;
  if (str == "ratio_scale_interval") return FeatureTypes::RATIO_SCALE_INTERVAL;
  if (str == "categorical_nominal") return FeatureTypes::CATEGORICAL_NOMINAL;
  if (str == "categorical_ordinal") return FeatureTypes::CATEGORICAL_ORDINAL;
  if (str == "binary_symmetric") return FeatureTypes::BINARY_SYMMETRIC;
  if (str == "binary_asymmetric") return FeatureTypes::BINARY_ASYMMETRIC;

  throw std::invalid_argument("Invalid feature type: " + str);
}

inline auto string_to_missing_strategy(const std::string& str)
    -> MissingStrategy {
  if (str == "ignore") return MissingStrategy::IGNORE;
  if (str == "max_dist") return MissingStrategy::MAX_DISTANCE;
  if (str == "raise_error") return MissingStrategy::RAISE_ERROR;

  throw std::invalid_argument("Invalid missing strategy: " + str);
}

inline auto string_to_scale_method(const std::string& str) -> ScaleMethods {
  if (str == "range") return ScaleMethods::RANGE;
  if (str == "iqr") return ScaleMethods::IQR;

  throw std::invalid_argument("Invalid scale method: " + str);
}

inline auto string_to_categorical_ordinal_calculation_type(
    const std::string& str) -> CategoricalOrdinalCalculationTypes {
  if (str == "kaufman") return CategoricalOrdinalCalculationTypes::KAUFMAN;
  if (str == "podani") return CategoricalOrdinalCalculationTypes::PODANI;

  throw std::invalid_argument("Invalid categorical ordinal calculation type: " +
                              str);
}

inline auto string_to_discretization(const std::string& str) -> Discretization {
  if (str.empty()) return Discretization::NONE;
  if (str == "silverman") return Discretization::SILVERMAN;
  if (str == "knn") return Discretization::KNN;

  throw std::invalid_argument("Invalid discretization: " + str);
}

template <typename E>
inline auto string_to_handle_unseen(const std::string& str) -> E {
  if (str == "warning") return E::WARNING;
  if (str == "error") return E::ERROR;
  if (str == "missing") return E::MISSING;

  throw std::invalid_argument("Invalid handle-unseen strategy: " + str);
}

inline auto string_to_out_of_range_strategy(const std::string& str)
    -> OutOfRangeStrategy {
  if (str == "clip") return OutOfRangeStrategy::CLIP;
  if (str == "warning") return OutOfRangeStrategy::WARNING;
  if (str == "error") return OutOfRangeStrategy::ERROR;

  throw std::invalid_argument("Invalid out-of-range strategy: " + str);
}

// used during conditional distances calculations and during CppConfig init
inline auto count_as_categorical(const FeatureTypes ft) -> bool {
  return ft == FeatureTypes::BINARY_SYMMETRIC ||
         ft == FeatureTypes::BINARY_ASYMMETRIC ||
         ft == FeatureTypes::CATEGORICAL_NOMINAL ||
         ft == FeatureTypes::CATEGORICAL_ORDINAL;
}

// EOF
