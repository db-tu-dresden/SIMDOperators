#include <catch2/catch_all.hpp>

#include "SIMDOperators/utils/AlignmentHelper.hpp"
#include <SIMDOperators/SIMDOperators.h>
#include <iostream>
#include <tslintrin.hpp>
#include <SIMDOperators/wrappers/DAPHNE/projectionPath.hpp>


using namespace std;
using namespace tuddbs;


TEST_CASE("Test ProjectionPath - varying project operations and different vector extensions", "[operators]"){
    

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

        auto pos_1 = Column<uint64_t>::create(20, ps::vector_size_B());
        pos_1->setPopulationCount(20);
        // fill column
        {
            auto data = pos_1->getData();
            for (int i = 0; i < pos_1->getLength(); ++i) {
                data[i] = i*5;
            }
        }

        auto pos_2 = Column<uint64_t>::create(5, ps::vector_size_B());
        pos_2->setPopulationCount(5);
        // fill column
        {
            auto data = pos_2->getData();
            for (int i = 0; i < pos_2->getLength(); ++i) {
                data[i] = i;
            }
        }

        // test normal project
        {
            auto proj_res = tuddbs::projectionPath<ps>::apply(col, pos_2, pos_1);
            // check the result
            CHECK(proj_res->getPopulationCount() == 5);
            auto data = proj_res->getData();
            for (int i = 0; i < proj_res->getPopulationCount(); ++i) {
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
            auto data = col->getData();
            for (int i = 0; i < col->getLength(); ++i) {
                data[i] = i;
            }
        }

        auto pos = Column<uint64_t>::create(20, ps::vector_size_B());
        pos->setPopulationCount(20);
        // fill column
        {
            auto data = pos->getData();
            for (int i = 0; i < pos->getLength(); ++i) {
                data[i] = i*5;
            }
        }

        // test normal project
        {
            auto proj_res = tuddbs::project<ps>::apply(col, pos);
            // check the result
            CHECK(proj_res->getPopulationCount() == 20);
            auto data = proj_res->getData();
            for (int i = 0; i < proj_res->getPopulationCount(); ++i) {
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
            auto data = col->getData();
            for (int i = 0; i < col->getLength(); ++i) {
                data[i] = i;
            }
        }

        // test with greater_than
        {
            auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);
            // check the result
            CHECK(select_res.get()->getPopulationCount() == 49);
            auto data = select_res->getData();
            for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                CHECK(data[i] == i + 51);
            }
        }

        // test with less_than
        {
            auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col, 50);
            // check the result
            CHECK(select_res->getPopulationCount() == 50);
            auto data = select_res->getData();
            for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                CHECK(data[i] == i);
            }
        }

        // test with equal
        {
            auto select_res = tuddbs::select<ps, tsl::functors::equal>::apply(col, 50);
            // check the result
            CHECK(select_res->getPopulationCount() == 1);
            auto data = select_res->getData();
            for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                CHECK(data[i] == 50);
            }
        }
    }
    */
}

TEST_CASE("Test ProjectionPath - Unaligned column test"){
    using ps = typename tsl::simd<uint64_t, tsl::sse>;

    auto col = Column<uint64_t>::create(100, ps::vector_size_B());
    col->setPopulationCount(100);
    // fill column
    {
        auto data = col->getData();
        for (int i = 0; i < col->getLength(); ++i) {
            data[i] = i;
        }
    }

    auto pos = Column<uint64_t>::create(20, ps::vector_size_B());
    pos->setPopulationCount(20);
    // fill column
    {
        auto data = pos->getData();
        for (int i = 0; i < pos->getLength(); ++i) {
            data[i] = i*5;
        }
    }


    {
        size_t offset = 3;
        /// Create a new column with an unaligned pointer
        auto pos_unaligned = pos->chunk(offset);
        auto project_res = tuddbs::project<ps>::apply(col, pos_unaligned);
        // check the result
        CHECK(project_res->getPopulationCount() == 20-offset);
        auto data = project_res->getData();
        for (int i = 0; i < project_res->getPopulationCount(); ++i) {
            CHECK(data[i] == offset*5 + i*5);
        }
    }
    {
        size_t offset = 7;
        /// Create a new column with an unaligned pointer
        auto pos_unaligned = pos->chunk(offset);
        auto project_res = tuddbs::project<ps>::apply(col, pos_unaligned);
        // check the result
        CHECK(project_res->getPopulationCount() == 20-offset);
        auto data = project_res->getData();
        for (int i = 0; i < project_res->getPopulationCount(); ++i) {
            CHECK(data[i] == offset*5 + i*5);
        }
    } 
    {
        size_t offset = 13;
        /// Create a new column with an unaligned pointer
        auto pos_unaligned = pos->chunk(offset);
        auto project_res = tuddbs::project<ps>::apply(col, pos_unaligned);
        // check the result
        CHECK(project_res->getPopulationCount() == 20-offset);
        auto data = project_res->getData();
        for (int i = 0; i < project_res->getPopulationCount(); ++i) {
            CHECK(data[i] == offset*5 + i*5);
        }
    }

}
