#pragma once

template <typename T>
struct NumericParams {
  const T* x;
  const T* y;
  const T* ranges;
  const T* bandwidths;
  int i;
};

template <typename T>
auto compute_numeric_diff(const NumericParams<T>& params) -> T {
  const int i = params.i;
  const T d = params.x[i] - params.y[i];
  T abs_diff = d < T(0) ? -d : d;

  if (params.ranges[i] > 0) {
    T diff = abs_diff / params.ranges[i];

    if (params.bandwidths[i] > 0 && abs_diff <= params.bandwidths[i]) {
      return T(0.0);
    }

    if (diff > T(1.0)) return T(1.0);
    return diff;
  }

  return T(0.0);
}

// EOF
