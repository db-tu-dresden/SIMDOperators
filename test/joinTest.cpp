#include <catch2/catch_all.hpp>

#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/SIMDOperators.h>
#include <iostream>
#include <tslintrin.hpp>



using namespace std;
using namespace tuddbs;


TEST_CASE("Test Join - varying compare opeartions and different vector extensions", "[operators]"){
    

    SECTION("using AVX512"){
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;

        auto col = Column<uint64_t>::create(100, ps::vector_size_B());
        col->setPopulationCount(100);
        // fill column
        {
            auto data = col->getData();
            for (int i = 0; i < col->getLength(); ++i) {
                data[i] = i;
            }
        }

        auto col2 = Column<uint64_t>::create(50, ps::vector_size_B());
        col2->setPopulationCount(50);
        // fill column
        {
            auto data = col2->getData();
            for (int i = 0; i < col2->getLength(); ++i) {
                data[i] = i*2;
            }
        }

        {
            auto join_res = tuddbs::natural_equi_join<ps>(col, col2);
            // check the result lhs
            CHECK(std::get<0>(join_res)->getPopulationCount() == 50);
            auto data_lhs = std::get<0>(join_res)->getData();
            for (int i = 0; i < std::get<0>(join_res)->getPopulationCount(); ++i) {
                CHECK(data_lhs[i] == i*2);
            }
            // check the result rhs
            CHECK(std::get<1>(join_res)->getPopulationCount() == 50);
            auto data_rhs = std::get<1>(join_res)->getData();
            for (int i = 0; i < std::get<1>(join_res)->getPopulationCount(); ++i) {
                CHECK(data_rhs[i] == i);
            }
        }

    }

    SECTION("using Scalar"){
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;

        auto col = Column<uint64_t>::create(100, ps::vector_size_B());
        col->setPopulationCount(100);
        // fill column
        {
            auto data = col->getData();
            for (int i = 0; i < col->getLength(); ++i) {
                data[i] = i;
            }
        }

        auto col2 = Column<uint64_t>::create(50, ps::vector_size_B());
        col2->setPopulationCount(50);
        // fill column
        {
            auto data = col2->getData();
            for (int i = 0; i < col2->getLength(); ++i) {
                data[i] = i*2;
            }
        }

        {
            auto join_res = tuddbs::natural_equi_join<ps>(col, col2);
            // check the result lhs
            CHECK(std::get<0>(join_res)->getPopulationCount() == 50);
            auto data_lhs = std::get<0>(join_res)->getData();
            for (int i = 0; i < std::get<0>(join_res)->getPopulationCount(); ++i) {
                CHECK(data_lhs[i] == i*2);
            }
            // check the result rhs
            CHECK(std::get<1>(join_res)->getPopulationCount() == 50);
            auto data_rhs = std::get<1>(join_res)->getData();
            for (int i = 0; i < std::get<1>(join_res)->getPopulationCount(); ++i) {
                CHECK(data_rhs[i] == i);
            }
        }
    }

    /*
    SECTION("using SSE"){
        using ps = typename tsl::simd<uint64_t, tsl::sse>;

        auto col = Column<uint64_t>::create(100, ps::vector_size_B());
        col->setPopulationCount(100);
        // fill column
        {
            auto data = col.get()->getData();
            for (int i = 0; i < col.get()->getLength(); ++i) {
                data[i] = i;
            }
        }

        // test with greater_than
        {
            auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);
            // check the result
            CHECK(select_res.get()->getPopulationCount() == 49);
            auto data = select_res.get()->getData();
            for (int i = 0; i < select_res.get()->getPopulationCount(); ++i) {
                CHECK(data[i] == i + 51);
            }
        }

        // test with less_than
        {
            auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col, 50);
            // check the result
            CHECK(select_res.get()->getPopulationCount() == 50);
            auto data = select_res.get()->getData();
            for (int i = 0; i < select_res.get()->getPopulationCount(); ++i) {
                CHECK(data[i] == i);
            }
        }

        // test with equal
        {
            auto select_res = tuddbs::select<ps, tsl::functors::equal>::apply(col, 50);
            // check the result
            CHECK(select_res.get()->getPopulationCount() == 1);
            auto data = select_res.get()->getData();
            for (int i = 0; i < select_res.get()->getPopulationCount(); ++i) {
                CHECK(data[i] == 50);
            }
        }
    }
    /**/
}

TEST_CASE("Test Join - Unaligned columns test"){
    using ps = typename tsl::simd<uint64_t, tsl::avx512>;

    auto col = Column<uint64_t>::create(100, ps::vector_size_B());
    col->setPopulationCount(100);
    // fill column
    {
        auto data = col->getData();
        for (int i = 0; i < col->getLength(); ++i) {
            data[i] = i;
        }
    }

    auto col2 = Column<uint64_t>::create(50, ps::vector_size_B());
    col2->setPopulationCount(50);
    // fill column
    {
        auto data = col2->getData();
        for (int i = 0; i < col2->getLength(); ++i) {
            data[i] = i*2;
        }
    }

    {
        size_t offset = 3;
        /// Create new columns with unaligned pointers
        auto col_unaligned = col->chunk(offset);
        auto col_unaligned2 = col2->chunk(offset);
        auto join_res = tuddbs::natural_equi_join<ps>(col_unaligned, col_unaligned2);
        // check the result lhs
        CHECK(std::get<0>(join_res)->getPopulationCount() == 47);
        auto data_lhs = std::get<0>(join_res)->getData();
        for (int i = 3; i < std::get<0>(join_res)->getPopulationCount(); ++i) {
            CHECK(data_lhs[i] == i*2+3);
        }
        // check the result rhs
        CHECK(std::get<1>(join_res)->getPopulationCount() == 47);
        auto data_rhs = std::get<1>(join_res)->getData();
        for (int i = 3; i < std::get<1>(join_res)->getPopulationCount(); ++i) {
            CHECK(data_rhs[i] == i);
        }
    }
    {
        size_t offset = 7;
        /// Create new columns with unaligned pointers
        auto col_unaligned = col->chunk(offset);
        auto col_unaligned2 = col2->chunk(offset);
        auto join_res = tuddbs::natural_equi_join<ps>(col_unaligned, col_unaligned2);
        // check the result lhs
        CHECK(std::get<0>(join_res)->getPopulationCount() == 43);
        auto data_lhs = std::get<0>(join_res)->getData();
        for (int i = 7; i < std::get<0>(join_res)->getPopulationCount(); ++i) {
            CHECK(data_lhs[i] == i*2+7);
        }
        // check the result rhs
        CHECK(std::get<1>(join_res)->getPopulationCount() == 43);
        auto data_rhs = std::get<1>(join_res)->getData();
        for (int i = 7; i < std::get<1>(join_res)->getPopulationCount(); ++i) {
            CHECK(data_rhs[i] == i);
        }
    }
    {
        size_t offset = 27;
        /// Create new columns with unaligned pointers
        auto col_unaligned = col->chunk(offset);
        auto col_unaligned2 = col2->chunk(offset);
        auto join_res = tuddbs::natural_equi_join<ps>(col_unaligned, col_unaligned2);
        // check the result lhs
        CHECK(std::get<0>(join_res)->getPopulationCount() == 23);
        auto data_lhs = std::get<0>(join_res)->getData();
        for (int i = 27; i < std::get<0>(join_res)->getPopulationCount(); ++i) {
            CHECK(data_lhs[i] == i*2+27);
        }
        // check the result rhs
        CHECK(std::get<1>(join_res)->getPopulationCount() == 23);
        auto data_rhs = std::get<1>(join_res)->getData();
        for (int i = 27; i < std::get<1>(join_res)->getPopulationCount(); ++i) {
            CHECK(data_rhs[i] == i);
        }
    }



};
