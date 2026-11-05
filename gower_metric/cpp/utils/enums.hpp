// Copyright (c) 2025 - 2026 the gower-metric developers
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

enum class FeatureTypes : std::uint8_t {
  CATEGORICAL_ORDINAL,
  CATEGORICAL_NOMINAL,
  BINARY_SYMMETRIC,
  BINARY_ASYMMETRIC,
  NUMERIC,
  RATIO_SCALE_INTERVAL,
};

enum class MissingStrategy : std::uint8_t {
  IGNORE,
  MAX_DISTANCE,
  RAISE_ERROR,
};

enum class CategoricalOrdinalCalculationTypes : std::uint8_t {
  KAUFMAN,
  PODANI,
};

enum class UnseenStrategy : std::uint8_t {
  WARNING,
  ERROR,
  MISSING,
};

enum class OutOfRangeStrategy : std::uint8_t {
  CLIP,
  WARNING,
  ERROR,
};

// EOF