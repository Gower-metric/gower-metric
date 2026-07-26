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

enum class ScaleMethods : std::uint8_t {
  RANGE,
  IQR,
};

enum class Discretization : std::uint8_t {
  NONE,
  SILVERMAN,
  KNN,
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

enum class HandleUnseenBinaryAsymmetric : std::uint8_t {
  WARNING,
  ERROR,
  MISSING,
};

enum class HandleUnseenBinarySymmetric : std::uint8_t {
  WARNING,
  ERROR,
  MISSING,
};

enum class HandleUnseenCategoricalNominal : std::uint8_t {
  WARNING,
  ERROR,
  MISSING,
};

enum class HandleUnseenCategoricalOrdinal : std::uint8_t {
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