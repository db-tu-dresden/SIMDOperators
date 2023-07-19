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


TEST_CASE("Test Calc - varying calc operations and different vector extrensions", "[operators]"){

    SECTION("Using Scalar"){
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
        // check for required primitives
        if( !tuddbs::calc_binary_core<ps,tsl::functors::add>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
        
            auto colLhs = new Column<uint64_t>(100, ps::vector_size_B());
            auto colRhs = new Column<uint64_t>(100, ps::vector_size_B());
            colLhs->setPopulationCount(100);
            colRhs->setPopulationCount(100);
            // fill column
            {
                auto dataLhs = colLhs->getData();
                auto dataRhs = colRhs->getData();
                for (int i = 0; i < colLhs->getLength(); ++i) {
                    dataLhs[i] = i;
                    dataRhs[i] = i;
                }
            }


            
            // test with add
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::add<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::add>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i*2);
                }
            } else {
                SKIP("Primitive [add] for [Scalar] not implemented or available on your system.");
            }
            
            // test with sub
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::sub<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::sub>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 0);
                }
            } else {
                SKIP("Primitive [sub] for [Scalar] not implemented or available on your system.");
            }
            
            // test with mul
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::mul<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::mul>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i*i);
                }
            } else {
                SKIP("Primitive [mul] for [Scalar] not implemented or available on your system.");
            }
        }
    }
    
    SECTION("Using AVX512"){
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;
        // check for required primitives
        if( !tuddbs::calc_binary_core<ps,tsl::functors::add>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
        
            auto colLhs = new Column<uint64_t>(100, ps::vector_size_B());
            auto colRhs = new Column<uint64_t>(100, ps::vector_size_B());
            colLhs->setPopulationCount(100);
            colRhs->setPopulationCount(100);
            // fill column
            {
                auto dataLhs = colLhs->getData();
                auto dataRhs = colRhs->getData();
                for (int i = 0; i < colLhs->getLength(); ++i) {
                    dataLhs[i] = i;
                    dataRhs[i] = i;
                }
            }


            
            // test with add
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::add<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::add>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i*2);
                }
            } else {
                SKIP("Primitive [add] for [avx512] not implemented or available on your system.");
            }
            
            // test with sub
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::sub<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::sub>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 0);
                }
            } else {
                SKIP("Primitive [sub] for [avx512] not implemented or available on your system.");
            }
            


            // test with mul
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::mul<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::mul>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i*i);
                }
            } else {
                SKIP("Primitive [mul] for [avx512] not implemented or available on your system.");
            }
        }
    }
    
    SECTION("Using SSE"){
        using ps = typename tsl::simd<uint64_t, tsl::sse>;
        // check for required primitives
        if( !tuddbs::calc_binary_core<ps,tsl::functors::add>::is_available ){
            SKIP("Not all required primitives are available on your system.");
        } else {
        
            auto colLhs = new Column<uint64_t>(100, ps::vector_size_B());
            auto colRhs = new Column<uint64_t>(100, ps::vector_size_B());
            colLhs->setPopulationCount(100);
            colRhs->setPopulationCount(100);
            // fill column
            {
                auto dataLhs = colLhs->getData();
                auto dataRhs = colRhs->getData();
                for (int i = 0; i < colLhs->getLength(); ++i) {
                    dataLhs[i] = i;
                    dataRhs[i] = i;
                }
            }


            
            // test with add
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::add<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::add>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i*2);
                }
            } else {
                SKIP("Primitive [add] for [sse] not implemented or available on your system.");
            }
            
            // test with sub
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::sub<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::sub>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == 0);
                }
            } else {
                SKIP("Primitive [sub] for [sse] not implemented or available on your system.");
            }
            


            // test with mul
            if constexpr ( detector::has_static_method_apply_v<tsl::functors::mul<ps, tsl::workaround>> )
            {
                auto calc_res = tuddbs::calc_binary<ps, tsl::functors::mul>::apply(colRhs, colLhs);
                // check the result
                CHECK(calc_res->getPopulationCount() == 100);
                auto data = calc_res->getData();
                for (int i = 0; i < calc_res->getPopulationCount(); ++i) {
                    CHECK(data[i] == i*i);
                }
            } else {
                SKIP("Primitive [mul] for [sse] not implemented or available on your system.");
            }
        }
    }

}

