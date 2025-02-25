#pragma once

#include <climits>
#include <bit>

namespace tuddbs {
  template <typename T>
  constexpr int log2_cexpr(T n, int p=0) {
    return (n<=1) ? p : log2_cexpr(n>>1, p+1);
  }

  template <typename T>
  consteval int sizeof_b() {
    return (sizeof(T)*CHAR_BIT);
  }

  template <typename T>
  constexpr size_t next_power_of_2(T value) {
    return std::bit_ceil<size_t>(static_cast<size_t>(value));
  }
}