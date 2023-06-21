#include <catch2/catch_all.hpp>

#include "SIMDOperators/utils/AlignmentHelper.hpp"
#include <SIMDOperators/SIMDOperators.h>
#include <iostream>
#include <tslintrin.hpp>



using namespace std;
using namespace tuddbs;


TEST_CASE("Test Project - varying project operations and different vector extensions", "[operators]"){
    

    SECTION("using AVX512"){
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;

        auto col = Column<uint64_t>::create(100, ps::vector_size_B());
        col->setPopulationCount(100);
        // fill column
        {
            auto data = col.get()->getData();
            for (int i = 0; i < col.get()->getLength(); ++i) {
                data[i] = i;
            }
        }

        auto pos = Column<uint64_t>::create(20, ps::vector_size_B());
        pos->setPopulationCount(20);
        // fill column
        {
            auto data = pos.get()->getData();
            for (int i = 0; i < pos.get()->getLength(); ++i) {
                data[i] = i*5;
            }
        }

        // test normal project
        {
            auto proj_res = tuddbs::project<ps>::apply(col, pos);
            // check the result
            CHECK(proj_res.get()->getPopulationCount() == 20);
            auto data = proj_res.get()->getData();
            for (int i = 0; i < proj_res.get()->getPopulationCount(); ++i) {
                CHECK(data[i] == i*5);
            }
        }

    }

    SECTION("using Scalar"){
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;

        auto col = Column<uint64_t>::create(100, ps::vector_size_B());
        col->setPopulationCount(100);
        // fill column
        {
            auto data = col.get()->getData();
            for (int i = 0; i < col.get()->getLength(); ++i) {
                data[i] = i;
            }
        }

        auto pos = Column<uint64_t>::create(20, ps::vector_size_B());
        pos->setPopulationCount(20);
        // fill column
        {
            auto data = pos.get()->getData();
            for (int i = 0; i < pos.get()->getLength(); ++i) {
                data[i] = i*5;
            }
        }

        // test normal project
        {
            auto proj_res = tuddbs::project<ps>::apply(col, pos);
            // check the result
            CHECK(proj_res.get()->getPopulationCount() == 20);
            auto data = proj_res.get()->getData();
            for (int i = 0; i < proj_res.get()->getPopulationCount(); ++i) {
                CHECK(data[i] == i*5);
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

TEST_CASE("Test Project - Unaligned column test"){
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

    auto pos = Column<uint64_t>::create(20, ps::vector_size_B());
    pos->setPopulationCount(20);
    // fill column
    {
        auto data = pos.get()->getData();
        for (int i = 0; i < pos.get()->getLength(); ++i) {
            data[i] = i*5;
        }
    }


    {
        size_t offset = 3;
        /// Create a new column with an unaligned pointer
        auto pos_unaligned = pos->chunk(offset);
        auto project_res = tuddbs::project<ps>::apply(col, pos_unaligned);
        // check the result
        CHECK(project_res.get()->getPopulationCount() == 20-offset);
        auto data = project_res.get()->getData();
        for (int i = 0; i < project_res.get()->getPopulationCount(); ++i) {
            CHECK(data[i] == offset*5 + i*5);
        }
    }
    {
        size_t offset = 7;
        /// Create a new column with an unaligned pointer
        auto pos_unaligned = pos->chunk(offset);
        auto project_res = tuddbs::project<ps>::apply(col, pos_unaligned);
        // check the result
        CHECK(project_res.get()->getPopulationCount() == 20-offset);
        auto data = project_res.get()->getData();
        for (int i = 0; i < project_res.get()->getPopulationCount(); ++i) {
            CHECK(data[i] == offset*5 + i*5);
        }
    } 
    {
        size_t offset = 13;
        /// Create a new column with an unaligned pointer
        auto pos_unaligned = pos->chunk(offset);
        auto project_res = tuddbs::project<ps>::apply(col, pos_unaligned);
        // check the result
        CHECK(project_res.get()->getPopulationCount() == 20-offset);
        auto data = project_res.get()->getData();
        for (int i = 0; i < project_res.get()->getPopulationCount(); ++i) {
            CHECK(data[i] == offset*5 + i*5);
        }
    }



}
