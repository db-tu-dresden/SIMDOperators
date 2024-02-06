#include <iostream>

#include <SIMDOperators/SIMDOperators.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <tslintrin.hpp>

int main(){
    using namespace std;
    using namespace tuddbs;

    // project<int>::apply();
    // using ps = typename tsl::simd<uint64_t, tsl::avx512>;
    using ps = typename tsl::simd<uint64_t, tsl::avx512>;

    if(1){
        auto col = Column<uint64_t>::create(100, ps::vector_size_B());
        auto data = col->getRawDataPtr();

        for (int i = 0; i < col->getLength(); ++i) {
            auto alignment = AlignmentHelper<ps>::getAlignment(&data[i]);
            cout << alignment.getOffset() << " // " << alignment.getElementsUntilAlignment() << endl;
        }
    }


    if(0){
        Column<uint64_t> c{10, sizeof(uint64_t)};

        {
            auto data = c.getRawDataPtr();
            for (int i = 0; i < 10; ++i) {
                data[i] = i;
            }
        }

        {
            const auto col = c;
            auto data = col.getRawDataPtr();
            for (int i = 0; i < col.getLength(); ++i) {
                cout << data[i] << endl;
            }

            auto d2 = col.getRawDataPtr();
            for (int i = 0; i < 10; ++i) {
                cout << "alginment: " << (reinterpret_cast<size_t>(&d2[i]) % 64) << " vs " << AlignmentHelper<ps>::getAlignment(&d2[i]).getOffset() << endl;
            }
        }

        {
            auto col = c.chunk(2, 3);
            for (int i = 0; i < col->getLength(); ++i) {
                cout << col->getRawDataPtr()[i] << endl;
            }
        }

        auto c2 = new Column<uint64_t>(10, sizeof(uint64_t));
        {
            auto data = c2->getRawDataPtr();
            for (int i = 0; i < c2->getLength(); ++i) {
                data[i] = i;
            }

            auto d2 = c2->getRawDataPtr();
            for (int i = 0; i < c2->getLength(); ++i) {
                cout << d2[i] << endl;
            }

        }
        delete c2;
    }

    if(0){
        auto colPtr = Column<uint64_t>::create(100, ps::vector_size_B());
        auto col = colPtr.get();
        col->setPopulationCount(100);
        cout << col->getAlignment() << endl;
        // fill column
        {
            auto data = col->getRawDataPtr();
            for (int i = 0; i < col->getLength(); ++i) {
                data[i] = i;
            }
        }



        // print column
        {
            auto data = col->getRawDataPtr();
            for (int i = 0; i < col->getLength(); ++i) {
                cout << data[i] << endl;
            }
        }


        
        auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);

        // print select result
        {   
            cout << "select result" << endl;
            auto data = select_res->getRawDataPtr();
            for (int i = 0; i < select_res->getLength(); ++i) {
                cout << data[i] << endl;
            }
        }
    }



}