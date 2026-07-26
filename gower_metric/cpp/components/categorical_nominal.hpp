#pragma once

template <typename T>
struct CategoricalNominalParams {
  const T* x;
  const T* y;
  int i;
};

template <typename T>
auto compute_categorical_nominal_diff(const CategoricalNominalParams<T>& params)
    -> T {
  const int i = params.i;

  return params.x[i] == params.y[i] ? T(0) : T(1);
}

// EOF
