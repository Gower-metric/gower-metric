// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

// Half-precision (IEEE 754 binary16) support
// Sources:
// https://nanobind.readthedocs.io/en/latest/ndarray.html#nonstandard-arithmetic-types
// https://nanobind.readthedocs.io/en/latest/porting.html#type-casters

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

#if defined(__FLT16_MAX__)
#define GOWER_HAS_FLOAT16 1

using float16 = _Float16;

template <>
struct nanobind::detail::dtype_traits<float16> {
  static constexpr dlpack::dtype value{
      static_cast<uint8_t>(dlpack::dtype_code::Float),
      16,
      1,
  };
  static constexpr auto name = const_name("float16");
};

template <>
struct nanobind::detail::type_caster<float16> {
  // NOLINTNEXTLINE(modernize-use-trailing-return-type)
  NB_TYPE_CASTER(float16, const_name("float"))

  auto from_python(const handle src, const uint8_t flags,
                   cleanup_list* cleanup) noexcept -> bool {
    make_caster<double> caster;
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = static_cast<float16>(caster.value);
    return true;
  }

  static auto from_cpp(const float16 src, const rv_policy policy,
                       cleanup_list* cleanup) noexcept -> handle {
    return make_caster<double>::from_cpp(src, policy, cleanup);
  }
};

#endif  // __FLT16_MAX__

// EOF
