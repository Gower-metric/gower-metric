#include "gower_config.hpp"

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
  is_categorical_feature.assign(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    if (count_as_categorical(feature_types[i])) {
      is_categorical_feature[i] = 1;
      ++p_cat;
    }
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
  missing_strategy = string_to_missing_strategy(data.missing_strategy);
  scale_method = string_to_scale_method(data.scale_method);
  categorical_ordinal_calculation_type =
      string_to_categorical_ordinal_calculation_type(
          data.categorical_ordinal_calculation_type);
  conditional_distances = data.conditional_distances;
  conditional_distances_threshold_coeff =
      data.conditional_distances_threshold_coeff;
  discretization = string_to_discretization(data.discretization);
  silverman_constant = data.silverman_constant;
  k_neighbors = data.k_neighbors;
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
  out_of_range = string_to_out_of_range_strategy(data.out_of_range);
  skip_out_of_range_validation = data.skip_out_of_range_validation;
  categorical_ordinal_values_order = data.categorical_ordinal_values_order;
  binary_asymmetric_value_order = data.binary_asymmetric_value_order;
  binary_symmetric_value_order = data.binary_symmetric_value_order;

  ordinal_meta.assign(n, OrdinalRankMeta<T>{});
  const bool podani = categorical_ordinal_calculation_type ==
                      CategoricalOrdinalCalculationTypes::PODANI;
  for (const auto& [j, counts] : data.ordinal_counts) {
    if (j < 0 || static_cast<std::size_t>(j) >= n) {
      throw std::invalid_argument("ordinal_counts feature index " +
                                  std::to_string(j) + " out of range [0, " +
                                  std::to_string(n) + ")");
    }
    if (counts.empty()) {
      continue;
    }

    OrdinalRankMeta<T>& meta = ordinal_meta[j];
    meta.mid.resize(counts.size());
    for (std::size_t r = 0; r < counts.size(); ++r) {
      meta.mid[r] = static_cast<T>((counts[r] - 1.0) / 2.0);
    }
    meta.denom = static_cast<T>(counts.size() - 1);
    meta.podani_denom = meta.denom - meta.mid.front() - meta.mid.back();
    meta.use_podani = podani && meta.podani_denom > T(0);
  }
}

template struct CppConfig<double>;
template struct CppConfig<float>;
#if defined(__FLT16_MAX__)
template struct CppConfig<_Float16>;
#endif
