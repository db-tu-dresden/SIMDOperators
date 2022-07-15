#include <iostream>

#include <column_based/filter.hpp>
#include <column.hpp>
int main() {
  std::cout << "Hello, World!" << std::endl;
  using namespace tuddbs;
  using namespace tvl;
  column_t<uint32_t> c{100, 64};
//  auto x = select<functors::between_inclusive, column_t<uint32_t>, simd<uint32_t, avx2>>(c, std::make_tuple((uint32_t)3,(uint32_t)5));
  auto x = select<functors::equal, column_t<uint32_t>, simd<uint32_t, avx2, 256>>(c, std::make_tuple((uint32_t)5));
  return 0;
}
