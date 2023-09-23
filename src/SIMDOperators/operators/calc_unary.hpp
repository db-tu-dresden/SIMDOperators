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
#ifndef SRC_OPERATORS_CALC_UNARY_HPP
#define SRC_OPERATORS_CALC_UNARY_HPP

#include "/home/dertuchi/work/TSL/generated_tsl/generator_output/include/tslintrin.hpp"
#include <cassert>
#include <type_traits>

namespace tuddbs {
    
    template< typename ps , template <typename ...> typename Operator>
    class calc_unary{
        using base_t = typename ps::base_type;
        using mask_t = typename ps::imask_type;
        using reg_t = typename ps::register_type;

        public:
        struct State{
            base_t* result_ptr;
            base_t const* p_Data1Ptr;
            size_t p_CountData1;
            base_t const* p_Data2Ptr = nullptr;
            size_t p_CountData2 = 0;
        };

        void operator()(State& myState){
            size_t element_count = ps::vector_element_count();

            const base_t *start = myState.p_Data1Ptr;
            const base_t *end = myState.p_Data1Ptr + myState.p_CountData1;

            reg_t vec;
            if constexpr(std::is_scalar<decltype(Operator< ps, tsl::workaround>::apply(vec))>::value){
                while(myState.p_Data1Ptr <= (end - element_count)){
                    vec = tsl::loadu< ps >(myState.p_Data1Ptr);
                    *myState.result_ptr = Operator< ps, tsl::workaround>::apply(vec);
                    myState.p_Data1Ptr += element_count;
                }
            }else{
                while(myState.p_Data1Ptr <= (end - element_count)){
                    vec = tsl::loadu< ps >(myState.p_Data1Ptr);

                    reg_t result = Operator< ps, tsl::workaround>::apply(vec);
                    tsl::storeu< ps >(myState.result_ptr, result);

                    myState.result_ptr += element_count;
                    myState.p_Data1Ptr += element_count;
                }
            }
        };

        static void flush(State& myState){
            const base_t *end = myState.p_Data1Ptr + myState.p_CountData1;
            reg_t vec;
            while(myState.p_Data1Ptr <= end){
                if constexpr(std::is_scalar<decltype(Operator< ps, tsl::workaround>::apply(vec))>::value){
                    *myState.result_ptr = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.p_Data1Ptr++);
                }else{
                    *myState.result_ptr++ = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.p_Data1Ptr++);
                }
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_CALC_UNARY_HPP