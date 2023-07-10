#pragma once
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"

#include "hipSYCL/sycl/libkernel/detail/bit_cast.hpp"
#ifdef HIPSYCL_HAS_CONSTEXPR_BITCAST
#define HIPSYCL_CONSTEXPR constexpr
#else
#define HIPSYCL_CONSTEXPR static inline
#endif

namespace hipsycl::fp16 {

HIPSYCL_UNIVERSAL_TARGET // So that CUDA calls are possible
HIPSYCL_CONSTEXPR float fp32_from_bits(__hipsycl_uint32 w) {
  float result = 0;
  HIPSYCL_INPLACE_BIT_CAST(__hipsycl_uint32, float, w, result);
  return result;
}

HIPSYCL_UNIVERSAL_TARGET // So that CUDA calls are possible
HIPSYCL_CONSTEXPR __hipsycl_uint32 fp32_to_bits(float f) {
  __hipsycl_uint32 result = 0;
  HIPSYCL_INPLACE_BIT_CAST(float, __hipsycl_uint32, f, result);
  return result;
}

HIPSYCL_UNIVERSAL_TARGET // So that CUDA calls are possible
HIPSYCL_CONSTEXPR double fp64_from_bits(__hipsycl_uint64 w) {
  double result = 0;
  HIPSYCL_INPLACE_BIT_CAST(__hipsycl_uint64, double, w, result);
  return result;
}

HIPSYCL_UNIVERSAL_TARGET // So that CUDA calls are possible
HIPSYCL_CONSTEXPR __hipsycl_uint64 fp64_to_bits(double f) {
  __hipsycl_uint64 result = 0;
  HIPSYCL_INPLACE_BIT_CAST(double, __hipsycl_uint64, f, result);
  return result;
}

}

#undef HIPSYCL_CONSTEXPR
