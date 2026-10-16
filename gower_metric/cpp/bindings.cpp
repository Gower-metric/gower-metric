// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <sstream>
#include <type_traits>

#include "engine.hpp"
#include "matrix/matrix.hpp"
#include "utils/config_converter.hpp"
#include "utils/half_float.hpp"

// Source
// https://nanobind.readthedocs.io/en/latest/classes.html#classes

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
struct RegisterConfigParams {
  nb::module_& m;
  const char* config_name;
  const char* data_name;
};

template <typename T>
struct Unseen {
  std::map<int, std::vector<T>> fresh;
  std::vector<std::string> warnings;
  size_t outside = -1;

  void warn(const int depth) const {
    if (warnings.empty()) return;
    const nb::gil_scoped_acquire gil;
    for (const std::string& text : warnings) {
      if (PyErr_WarnEx(PyExc_UserWarning, text.c_str(), depth) < 0) {
        throw nb::python_error();
      }
    }
  }
};

template <typename T>
auto check_rows(const CppConfig<T>& cfg, const T* data, const size_t n_rows,
                std::vector<T>& fixed, Unseen<T>& unseen) -> const T* {
  const auto nf = static_cast<size_t>(cfg.n_features);
  const T inf = static_cast<T>(std::numeric_limits<double>::infinity());
  std::string outside;
  int unstorable = -1;
  for (const int f : cfg.numeric_group) {
    T lo = inf;
    T hi = -inf;
    for (size_t r = 0; r < n_rows; ++r) {
      const T v = data[(r * nf) + f];
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
    if (cfg.out_of_range != OutOfRangeStrategy::CLIP &&
        (lo < cfg.lower[f] || hi > cfg.upper[f])) {
      std::ostringstream text;
      text << '\n'
           << (cfg.feature_types[f] == FeatureTypes::NUMERIC ? "numeric"
                                                             : "ratio_scale")
           << " column " << f << ": values in [" << static_cast<double>(lo)
           << ", " << static_cast<double>(hi) << "], fitted range ["
           << static_cast<double>(cfg.lower[f]) << ", "
           << static_cast<double>(cfg.upper[f]) << "]";
      outside += text.str();
    }
    if (unstorable < 0 && (lo == -inf || hi == inf)) unstorable = f;
  }
  if (!outside.empty()) {
    const std::string head =
        "Out-of-range values detected in transform data (distances will be "
        "clipped to [0, 1]):";
    if (cfg.out_of_range == OutOfRangeStrategy::ERROR) {
      throw std::invalid_argument(head + outside);
    }
    if (unseen.outside == size_t(-1)) {
      unseen.outside = unseen.warnings.size();
      unseen.warnings.push_back(head);
    }
    unseen.warnings[unseen.outside] += outside;
  }
  if (unstorable >= 0) {
    throw std::invalid_argument(
        std::string(feature_type_name(cfg.feature_types[unstorable])) +
        " column " + std::to_string(unstorable) +
        " holds a value this precision cannot store.");
  }
  for (const int f : cfg.checked_features) {
    const FeatureDomain<T>& domain = cfg.domains[f];
    bool warned = false;
    for (size_t r = 0; r < n_rows; ++r) {
      const T v = data[(r * nf) + f];
      const size_t count = domain.values.size();
      if (!gower_detail::is_present(v) ||
          (domain.positions ? gower_detail::level_index(v, count) != count
                            : std::binary_search(domain.values.begin(),
                                                 domain.values.end(), v))) {
        continue;
      }
      std::ostringstream text;
      text << "Unseen value " << static_cast<double>(v) << " in "
           << feature_type_name(cfg.feature_types[f]) << " column " << f
           << ": it is not one of the values the column was fitted with.";
      bool refused = domain.strategy == UnseenStrategy::ERROR;
      if (domain.room <= 2) {
        std::vector<T>& seen = unseen.fresh[f];
        if (std::find(seen.begin(), seen.end(), v) == seen.end()) {
          seen.push_back(v);
        }
        if (seen.size() > domain.room) {
          refused = true;
          text << " A binary feature holds two values.";
        }
      }
      if (refused) throw std::invalid_argument(text.str());
      if (domain.strategy == UnseenStrategy::WARNING && !warned) {
        warned = true;
        unseen.warnings.push_back(text.str() +
                                  " Treating it as missing (np.nan).");
      }
      if (fixed.empty()) fixed.assign(data, data + (n_rows * nf));
      fixed[(r * nf) + f] =
          static_cast<T>(std::numeric_limits<double>::quiet_NaN());
    }
  }
  return fixed.empty() ? data : fixed.data();
}

template <typename T, typename TOut>
void register_matrix_methods(nb::class_<CppConfig<T>>& cls) {
  using Matrix =
      nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  using OutMatrix =
      nb::ndarray<TOut, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  using OutVector =
      nb::ndarray<TOut, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  cls.def(
      "calculate_matrix",
      [](const CppConfig<T>& self, const Matrix& data, const OutMatrix& out,
         const bool similarity) -> void {
        nb::gil_scoped_release release;
        const size_t n = data.shape(0);
        if (data.shape(1) != static_cast<size_t>(self.n_features)) {
          throw std::invalid_argument("data must have n_features columns");
        }
        if (out.shape(0) != n || out.shape(1) != n) {
          throw std::invalid_argument("out must have shape (n_rows, n_rows)");
        }
        std::vector<T> fixed;
        Unseen<T> unseen;
        const T* rows = check_rows(self, data.data(), n, fixed, unseen);
        unseen.warn(4);
        compute_gower_matrix<T, TOut>(self, rows, out.data(), n, similarity);
      },
      "data"_a.noconvert(), "out"_a.noconvert(), "similarity"_a = false,
      "Fill a pre-allocated (n, n) matrix with pairwise Gower distances. "
      "`out` may be any supported float width; narrower than the model's own "
      "precision rounds each cell on the way out.");

  cls.def(
      "calculate_matrix_condensed",
      [](const CppConfig<T>& self, const Matrix& data, const OutVector& out,
         const bool similarity) -> void {
        nb::gil_scoped_release release;
        const size_t n = data.shape(0);
        if (data.shape(1) != static_cast<size_t>(self.n_features)) {
          throw std::invalid_argument("data must have n_features columns");
        }
        const size_t expected = n < 2 ? 0 : n * (n - 1) / 2;
        if (out.shape(0) != expected) {
          throw std::invalid_argument("out must have length n * (n - 1) / 2");
        }
        std::vector<T> fixed;
        Unseen<T> unseen;
        const T* rows = check_rows(self, data.data(), n, fixed, unseen);
        unseen.warn(4);
        compute_gower_matrix_condensed<T, TOut>(self, rows, out.data(), n,
                                                similarity);
      },
      "data"_a.noconvert(), "out"_a.noconvert(), "similarity"_a = false,
      "Fill a pre-allocated n*(n-1)/2 vector with the upper triangle of "
      "the pairwise Gower distances, in scipy pdist order.");

  cls.def(
      "calculate_matrix_xy",
      [](const CppConfig<T>& self, const Matrix& x_data, const Matrix& y_data,
         const OutMatrix& out, const bool similarity) -> void {
        nb::gil_scoped_release release;
        const size_t n_x = x_data.shape(0);
        const size_t n_y = y_data.shape(0);
        if (x_data.shape(1) != static_cast<size_t>(self.n_features) ||
            y_data.shape(1) != static_cast<size_t>(self.n_features)) {
          throw std::invalid_argument("x and y must have n_features columns");
        }
        if (out.shape(0) != n_x || out.shape(1) != n_y) {
          throw std::invalid_argument("out must have shape (n_x, n_y)");
        }
        std::vector<T> fixed_x;
        std::vector<T> fixed_y;
        Unseen<T> unseen;
        const T* x_rows = check_rows(self, x_data.data(), n_x, fixed_x, unseen);
        unseen.fresh.clear();
        const T* y_rows = check_rows(self, y_data.data(), n_y, fixed_y, unseen);
        unseen.warn(4);
        compute_gower_matrix_xy<T, TOut>(self, x_rows, n_x, y_rows, n_y,
                                         out.data(), similarity);
      },
      "x_data"_a.noconvert(), "y_data"_a.noconvert(), "out"_a.noconvert(),
      "similarity"_a = false,
      "Fill a pre-allocated (n_x, n_y) matrix with cross pairwise Gower.");
}

template <typename T>
void register_matrix_overloads(nb::class_<CppConfig<T>>& cls) {
  register_matrix_methods<T, T>(cls);
  if constexpr (!std::is_same_v<T, double>) {
    register_matrix_methods<T, double>(cls);
  }
  if constexpr (!std::is_same_v<T, float>) {
    register_matrix_methods<T, float>(cls);
  }
#ifdef GOWER_HAS_FLOAT16
  if constexpr (!std::is_same_v<T, float16>) {
    register_matrix_methods<T, float16>(cls);
  }
#endif
}

template <typename T>
void register_config(const RegisterConfigParams<T>& params) {
  using Row = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  nb::class_<CppConfigData<T>>(params.m, params.data_name)
      .def(nb::init<>())
      .def_rw("feature_types", &CppConfigData<T>::feature_types)
      .def_rw("feature_weights", &CppConfigData<T>::feature_weights)
      .def_rw("ranges", &CppConfigData<T>::ranges)
      .def_rw("bandwidths", &CppConfigData<T>::bandwidths)
      .def_rw("lower", &CppConfigData<T>::lower)
      .def_rw("upper", &CppConfigData<T>::upper)
      .def_rw("out_of_range", &CppConfigData<T>::out_of_range)
      .def_rw("missing_strategy", &CppConfigData<T>::missing_strategy)
      .def_rw("categorical_ordinal_calculation_type",
              &CppConfigData<T>::categorical_ordinal_calculation_type)
      .def_rw("conditional_distances", &CppConfigData<T>::conditional_distances)
      .def_rw("conditional_distances_threshold_coeff",
              &CppConfigData<T>::conditional_distances_threshold_coeff)
      .def_rw("handle_unseen_binary_asymmetric",
              &CppConfigData<T>::handle_unseen_binary_asymmetric)
      .def_rw("handle_unseen_binary_symmetric",
              &CppConfigData<T>::handle_unseen_binary_symmetric)
      .def_rw("handle_unseen_categorical_nominal",
              &CppConfigData<T>::handle_unseen_categorical_nominal)
      .def_rw("handle_unseen_categorical_ordinal",
              &CppConfigData<T>::handle_unseen_categorical_ordinal)
      .def_rw("ordinal_counts", &CppConfigData<T>::ordinal_counts)
      .def_rw("excluded_features", &CppConfigData<T>::excluded_features)
      .def_rw("category_values", &CppConfigData<T>::category_values);

  auto config =
      nb::class_<CppConfig<T>>(params.m, params.config_name)
          .def(nb::init<>())
          .def("configure_arguments", &CppConfig<T>::configure_arguments,
               "data"_a, nb::call_guard<nb::gil_scoped_release>(),
               "Populate the config from a ConfigData.")
          .def_ro("n_features", &CppConfig<T>::n_features)
          .def_ro("feature_weights", &CppConfig<T>::feature_weights)
          .def_ro("ranges", &CppConfig<T>::ranges)
          .def_ro("bandwidths", &CppConfig<T>::bandwidths)
          .def(
              "calculate_distance",
              [](const CppConfig<T>& self, const Row& x, const Row& y) -> T {
                if (x.shape(0) != static_cast<size_t>(self.n_features) ||
                    y.shape(0) != static_cast<size_t>(self.n_features)) {
                  throw std::invalid_argument(
                      "x and y must have length n_features");
                }
                std::vector<T> fixed_x;
                std::vector<T> fixed_y;
                Unseen<T> unseen;
                const T* a = check_rows(self, x.data(), 1, fixed_x, unseen);
                const T* b = check_rows(self, y.data(), 1, fixed_y, unseen);
                unseen.warn(2);
                return self.calculate_distance(a, b);
              },
              "x"_a.noconvert(), "y"_a.noconvert(),
              nb::call_guard<nb::gil_scoped_release>(),
              "Gower distance between two 1-D feature vectors.");

  register_matrix_overloads<T>(config);
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,modernize-use-trailing-return-type)
NB_MODULE(cpp_engine, m) {
  m.doc() = "Gower metric C++ core (nanobind bindings)";

  register_config<double>(
      RegisterConfigParams<double>{m, "CppConfig", "CppConfigData"});
  register_config<float>(
      RegisterConfigParams<float>{m, "CppConfigF", "CppConfigDataF"});
#ifdef GOWER_HAS_FLOAT16
  register_config<float16>(
      RegisterConfigParams<float16>{m, "CppConfigH", "CppConfigDataH"});
#endif
}
