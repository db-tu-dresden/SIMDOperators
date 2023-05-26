#include "utils/AlignmentHelper.hpp"
#include <iostream>
#include <tslintrin.hpp>
#include <SIMDOperators.h>

int main(){
    using namespace std;
    using namespace tuddbs;

    // project<int>::apply();
    // using ps = typename tsl::simd<uint64_t, tsl::avx512>;
    using ps = typename tsl::simd<uint64_t, tsl::avx512>;

    Column<uint64_t> c{10, sizeof(uint64_t)};

    {
        auto data = c.getData();
        for (int i = 0; i < 10; ++i) {
            data[i] = i;
        }
    }

    {
        const auto col = c;
        auto data = col.getData();
        for (int i = 0; i < col.getElementCount(); ++i) {
            cout << data[i] << endl;
        }

        auto d2 = col.getData().get();
        for (int i = 0; i < 10; ++i) {
            cout << "alginment: " << (reinterpret_cast<size_t>(&d2[i]) % 64) << " vs " << AlignmentHelper<ps>::getAlignment(&d2[i]).getOffset() << endl;
        }
    }

    {
        auto col = c.chunk(2, 3);
        for (int i = 0; i < col->getElementCount(); ++i) {
            cout << col->getData()[i] << endl;
        }
    }

    auto c2 = new Column<uint64_t>(10, sizeof(uint64_t));
    {
        auto data = c2->getData();
        for (int i = 0; i < c2->getElementCount(); ++i) {
            data[i] = i;
        }

        auto d2 = c2->getData().get();
        for (int i = 0; i < c2->getElementCount(); ++i) {
            cout << d2[i] << endl;
        }

    }
    delete c2;
}