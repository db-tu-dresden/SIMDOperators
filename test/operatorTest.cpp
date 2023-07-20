#include <catch2/catch_all.hpp>

#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/SIMDOperators.h>
#include <iostream>
#include <tslintrin.hpp>



using namespace std;
using namespace tuddbs;


TEST_CASE("Test Select - varying compare opeartions and different vector extensions", "[operators]"){
    

    SECTION("using AVX512"){
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;

        auto col = new Column<uint64_t>(100, ps::vector_size_B());
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
            CHECK(select_res->getPopulationCount() == 49);
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
        delete col;
    }

    SECTION("using Scalar"){
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;

        auto col = new Column<uint64_t>(100, ps::vector_size_B());
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
            CHECK(select_res->getPopulationCount() == 49);
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
        delete col;
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

TEST_CASE("Test Select - Unaligned column test"){
    using ps = typename tsl::simd<uint64_t, tsl::avx512>;

    auto col = new Column<uint64_t>(100, ps::vector_size_B());
    col->setPopulationCount(100);
    // fill column
    {
        auto data = col->getData();
        for (int i = 0; i < col->getLength(); ++i) {
            data[i] = i;
        }
    }


    {
        size_t offset = 3;
        /// Create a new column with an unaligned pointer
        auto col_unaligned = col->chunk(offset);
        auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col_unaligned, 50);
        // check the result
        CHECK(select_res->getPopulationCount() == 50-offset);
        auto data = select_res->getData();
        for (int i = 0; i < select_res->getPopulationCount(); ++i) {
            CHECK(data[i] == i);
        }
    }
    {
        size_t offset = 7;
        /// Create a new column with an unaligned pointer
        auto col_unaligned = col->chunk(offset);
        auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col_unaligned, 50);
        // check the result
        CHECK(select_res->getPopulationCount() == 50-offset);
        auto data = select_res->getData();
        for (int i = 0; i < select_res->getPopulationCount(); ++i) {
            CHECK(data[i] == i);
        }
    }
    {
        size_t offset = 27;
        /// Create a new column with an unaligned pointer
        auto col_unaligned = col->chunk(offset);
        auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col_unaligned, 50);
        // check the result
        CHECK(select_res->getPopulationCount() == 50-offset);
        auto data = select_res->getData();
        for (int i = 0; i < select_res->getPopulationCount(); ++i) {
            CHECK(data[i] == i);
        }
    }

    delete col;

}

TEST_CASE("Test Aggregate"){
    using ps = typename tsl::simd<uint64_t, tsl::avx2>;
    auto const data_size = 4ULL * 1024 * 8 + 24;
    auto col = new Column<uint64_t>(data_size/sizeof(uint64_t), ps::vector_size_B());
    col->setPopulationCount(data_size/sizeof(uint64_t));
    // fill column
    {
        auto data = col->getData();
        for (int i = 0; i < col->getLength(); ++i) {
            data[i] = i;
        }
    }

    using op_t = tuddbs::aggregate<ps, 4096, tsl::functors::add, tsl::functors::hadd>;
    op_t::intermediate_state_t state(col->getData());
    op_t aggregate;
    for (size_t batch = 0; batch < data_size / 4096; ++batch) {
      aggregate(state);
      state.advance();
    }
    op_t::flush_state_t flush_state((data_size & 4095)>>3, state);
    aggregate(flush_state);

    uint64_t n = (col->getLenght()-1);
    uint64_t expected_result = (n * (n+1)) >> 1;
    std::cout << "Result: " << flush_state.result << std::endl;
    std::cout << "Should be: " << expected_result << std::endl;
    CHECK(flush_state.result == expected_result);
    delete col;

}

