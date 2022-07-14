#include <iostream>

#include <column_based/filter.hpp>
#include <column.hpp>
int main() {
  std::cout << "Hello, World!" << std::endl;
  using namespace tuddbs;
  using namespace tvl;
  column_t<int64_t> c{100, 64};
  auto x = select<between_inclusive_fn, column_t<int64_t>, simd<int64_t, avx2>>(c, std::make_tuple((int64_t)3,(int64_t)5));
  return 0;
}
