#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "engine.hpp"
#include "matrix/matrix.hpp"
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
void register_config(const RegisterConfigParams<T>& params) {
  using Row = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using Matrix =
      nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  using OutMatrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

  nb::class_<CppConfigData<T>>(params.m, params.data_name)
      .def(nb::init<>())
      .def_rw("feature_types", &CppConfigData<T>::feature_types)
      .def_rw("feature_weights", &CppConfigData<T>::feature_weights)
      .def_rw("ranges", &CppConfigData<T>::ranges)
      .def_rw("bandwidths", &CppConfigData<T>::bandwidths)
      .def_rw("missing_strategy", &CppConfigData<T>::missing_strategy)
      .def_rw("scale_method", &CppConfigData<T>::scale_method)
      .def_rw("categorical_ordinal_calculation_type",
              &CppConfigData<T>::categorical_ordinal_calculation_type)
      .def_rw("conditional_distances", &CppConfigData<T>::conditional_distances)
      .def_rw("conditional_distances_threshold_coeff",
              &CppConfigData<T>::conditional_distances_threshold_coeff)
      .def_rw("discretization", &CppConfigData<T>::discretization)
      .def_rw("silverman_constant", &CppConfigData<T>::silverman_constant)
      .def_rw("k_neighbors", &CppConfigData<T>::k_neighbors)
      .def_rw("handle_unseen_binary_asymmetric",
              &CppConfigData<T>::handle_unseen_binary_asymmetric)
      .def_rw("handle_unseen_binary_symmetric",
              &CppConfigData<T>::handle_unseen_binary_symmetric)
      .def_rw("handle_unseen_categorical_nominal",
              &CppConfigData<T>::handle_unseen_categorical_nominal)
      .def_rw("handle_unseen_categorical_ordinal",
              &CppConfigData<T>::handle_unseen_categorical_ordinal)
      .def_rw("out_of_range", &CppConfigData<T>::out_of_range)
      .def_rw("skip_out_of_range_validation",
              &CppConfigData<T>::skip_out_of_range_validation)
      .def_rw("categorical_ordinal_values_order",
              &CppConfigData<T>::categorical_ordinal_values_order)
      .def_rw("binary_asymmetric_value_order",
              &CppConfigData<T>::binary_asymmetric_value_order)
      .def_rw("binary_symmetric_value_order",
              &CppConfigData<T>::binary_symmetric_value_order)
      .def_rw("ordinal_counts", &CppConfigData<T>::ordinal_counts);

  nb::class_<CppConfig<T>>(params.m, params.config_name)
      .def(nb::init<>())
      .def("configure_arguments", &CppConfig<T>::configure_arguments, "data"_a,
           nb::lock_self(), nb::call_guard<nb::gil_scoped_release>(),
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
            return self.calculate_distance(x.data(), y.data());
          },
          "x"_a, "y"_a, nb::call_guard<nb::gil_scoped_release>(),
          "Gower distance between two 1-D feature vectors.")
      .def(
          "calculate_matrix",
          [](const CppConfig<T>& self, const Matrix& data, const OutMatrix& out,
             const bool similarity) -> void {
            const size_t n = data.shape(0);
            if (data.shape(1) != static_cast<size_t>(self.n_features)) {
              throw std::invalid_argument("data must have n_features columns");
            }
            if (out.shape(0) != n || out.shape(1) != n) {
              throw std::invalid_argument(
                  "out must have shape (n_rows, n_rows)");
            }
            nb::gil_scoped_release release;
            compute_gower_matrix<T>(self, data.data(), out.data(), n,
                                    similarity);
          },
          "data"_a, "out"_a, "similarity"_a = false,
          "Fill a pre-allocated (n, n) matrix with pairwise Gower distances.")
      .def(
          "calculate_matrix_xy",
          [](const CppConfig<T>& self, const Matrix& x_data,
             const Matrix& y_data, const OutMatrix& out,
             const bool similarity) -> void {
            const size_t n_x = x_data.shape(0);
            const size_t n_y = y_data.shape(0);
            if (x_data.shape(1) != static_cast<size_t>(self.n_features) ||
                y_data.shape(1) != static_cast<size_t>(self.n_features)) {
              throw std::invalid_argument(
                  "x and y must have n_features columns");
            }
            if (out.shape(0) != n_x || out.shape(1) != n_y) {
              throw std::invalid_argument("out must have shape (n_x, n_y)");
            }
            nb::gil_scoped_release release;
            compute_gower_matrix_xy<T>(self, x_data.data(), n_x, y_data.data(),
                                       n_y, out.data(), similarity);
          },
          "x_data"_a, "y_data"_a, "out"_a, "similarity"_a = false,
          "Fill a pre-allocated (n_x, n_y) matrix with cross pairwise Gower.");
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
