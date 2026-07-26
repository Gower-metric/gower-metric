#pragma once

template <typename T>
struct CategoricalOrdinalParams {
  const T* x;
  const T* y;
  int i;
  const T* mid;
  T denom;
  T podani_denom;
  bool use_podani;
};

template <typename T>
auto compute_categorical_ordinal_diff(const CategoricalOrdinalParams<T>& params)
    -> T {
  const int i = params.i;
  const T rx = params.x[i];
  const T ry = params.y[i];
  const T diff = rx < ry ? ry - rx : rx - ry;

  if (params.use_podani) {
    T dist = (diff - params.mid[static_cast<int>(rx)] -
              params.mid[static_cast<int>(ry)]) /
             params.podani_denom;

    if (dist < T(0)) return T(0);
    if (dist > T(1)) return T(1);
    return dist;
  }

  return params.denom > T(0) ? diff / params.denom : T(0);
}

// EOF
