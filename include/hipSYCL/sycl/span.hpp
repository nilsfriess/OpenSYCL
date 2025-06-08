#pragma once

#if defined __has_include
#if __has_include (<span>)
#include <span>

namespace hipsycl::sycl {
  template <typename T, std::size_t Extent = std::dynamic_extent>
  using span = std::span<T, Extent>;
}

#endif
#endif
