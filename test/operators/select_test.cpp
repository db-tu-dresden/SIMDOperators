// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Copyright (c) 2022 SimdOperators Team.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, version 3.
 
   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.
 
   You should have received a copy of the GNU General Public License 
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
// ------------------------------------------------------------------- //
#include <catch2/catch_all.hpp>

#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/SIMDOperators.h>
#include <SIMDOperators/utils/constexpr/MemberDetector.h>
#include <iostream>
#include <tslintrin.hpp>



using namespace std;
using namespace tuddbs;


TEST_CASE("Test Select - varying compare opeartions and different vector extensions", "[operators]"){
    

    SECTION("using AVX512"){
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;
        // check for required primitives
        if( !tuddbs::select_core<ps,tsl::functors::greater_than>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
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
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::greater_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 49);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i + 51);
                }
            } else {
                SKIP("Primitive [greater_than] for [avx512] not implemented or available on your system.");
            }

            // test with less_than
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 50);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i);
                }
            } else {
                SKIP("Primitive [less_than] for [avx512] not implemented or available on your system.");
            }

            // test with equal
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::equal<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::equal>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 1);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 50);
                }
            } else {
                SKIP("Primitive [equal] for [avx512] not implemented or available on your system.");
            }
            delete col;
        }
    }

    SECTION("using Scalar"){
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;

        // check for required primitives
        if( !tuddbs::select_core<ps,tsl::functors::greater_than>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
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
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::greater_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 49);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i + 51);
                }
            } else {
                SKIP("Primitive [greater_than] for [scalar] not implemented or available on your system.");
            }

            // test with less_than
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 50);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i);
                }
            } else {
                SKIP("Primitive [less_than] for [scalar] not implemented or available on your system.");
            }

            // test with equal
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::equal<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::equal>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 1);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 50);
                }
            } else {
                SKIP("Primitive [equal] for [scalar] not implemented or available on your system.");
            }
            delete col;
        }
    }

    
    SECTION("using SSE"){
        using ps = typename tsl::simd<uint64_t, tsl::sse>;

        // check for required primitives
        if constexpr ( !tuddbs::select_core<ps,tsl::functors::greater_than>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
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
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::greater_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 49);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i + 51);
                }
            } else {
                SKIP("Primitive [greater_than] for [sse] not implemented or available on your system.");
            }

            // test with less_than
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::less_than>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 50);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i);
                }
            } else {
                SKIP("Primitive [less_than] for [sse] not implemented or available on your system.");
            }

            // test with equal
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::equal<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::select<ps, tsl::functors::equal>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 1);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 50);
                }
            } else {
                SKIP("Primitive [equal] for [sse] not implemented or available on your system.");
            }
        }
    }
}

TEST_CASE("Test Select - Unaligned column test"){
    using ps = typename tsl::simd<uint64_t, tsl::avx512>;
    // check for required primitives
    if( !tuddbs::select_core<ps,tsl::functors::greater_than>::is_available ){
        SKIP("Not all required primitives are available on your system.");
    } else {

        auto col = new Column<uint64_t>(100, ps::vector_size_B());
        col->setPopulationCount(100);
        // fill column
        {
            auto data = col->getData();
            for (int i = 0; i < col->getLength(); ++i) {
                data[i] = i;
            }
        }


        if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
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
        } else {
            SKIP("Primitive [less_than] for [avx512] not implemented or available on your system.");
        }
        
        if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
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
        } else {
            SKIP("Primitive [less_than] for [avx512] not implemented or available on your system.");
        }

        if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
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
        } else {
            SKIP("Primitive [less_than] for [avx512] not implemented or available on your system.");
        }

        delete col;
    }
}

TEST_CASE("Test Select (MetaOperator) - varying compare opeartions and different vector extensions", "[operators]"){
    SECTION("using Scalar"){
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
        // check for required primitives
        if( !tuddbs::select_core<ps,tsl::functors::greater_than>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
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
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::greater_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::MetaOperator<ps, tuddbs::select_core<ps, tsl::functors::greater_than>>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 49);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i + 51);
                }
            } else {
                SKIP("Primitive [greater_than] for [scalar] not implemented or available on your system.");
            }

            // test with less_than
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::less_than<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::MetaOperator<ps, tuddbs::select_core<ps, tsl::functors::less_than>>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 50);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i);
                }
            } else {
                SKIP("Primitive [less_than] for [scalar] not implemented or available on your system.");
            }

            // test with equal
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::equal<ps, tsl::workaround>> )
            {
                auto select_res = tuddbs::MetaOperator<ps, tuddbs::select_core<ps, tsl::functors::equal>>::apply(col, 50);
                // check the result
                CHECK(select_res->getPopulationCount() == 1);
                auto data = select_res->getData();
                for (int i = 0; i < select_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 50);
                }
            } else {
                SKIP("Primitive [equal] for [scalar] not implemented or available on your system.");
            }
            delete col;
        }
    }
}
