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

namespace tuddbs {
    
    template< typename ps , template <typename ...> typename Operator>
    class calc_unary{
        using base_t = typename ps::base_type;
        using mask_t = typename ps::imask_type;
        using reg_t = typename ps::register_type;
        using offset_t = typename ps::offset_base_type;

        public:
        struct State{
            base_t* result_ptr;
            base_t const* data_ptr;
            size_t count;
        };

        struct State_gather{
            base_t* result_ptr;
            base_t const* data_ptr;
            offset_t* pos_ptr;
            size_t const pos_count;
        };

        struct State_bitlist{
            base_t* result_ptr;
            base_t const* data_ptr;
            mask_t bitlist_ptr;
            size_t count;
        };

        void operator()(State& myState){
            size_t element_count = ps::vector_element_count();

            const base_t *end = myState.data_ptr + myState.count;

            while(myState.data_ptr <= (end - element_count)){
                reg_t vec = tsl::loadu< ps >(myState.data_ptr);

                reg_t result = Operator< ps, tsl::workaround>::apply(vec);
                tsl::storeu< ps >(myState.result_ptr, result);

                myState.result_ptr += element_count;
                myState.data_ptr += element_count;
            }
        };

        void operator()(State_bitlist& myState){
            size_t const element_count = ps::vector_element_count();
            size_t const mask_size = sizeof(mask_t) * 8;

            const base_t *end = myState.data_ptr + myState.count;

            if constexpr(element_count < mask_size){
                // darauf achten, dass zuerst die relevanten bits verwendet werden (Stichwort: packed )
            }else{
                while(myState.data_ptr <= (end - element_count)){
                    base_t* data;
                    const base_t *start = data;
                    int mask_count;
                    while(mask_count < element_count){
                        mask_t mask = *myState.bitlist_ptr;
                        mask_count = tsl::mask_population_count< ps >(mask);

                        // Mask_count is greater than element count -> split mask so data fits again
                        if(mask_count > element_count){
                            // Find index where to split the mask
                            size_t index = 0;
                            mask_t split_mask = 0;
                            for (size_t i = 0; i < mask_size && index >= mask_count - element_count; i++) {
                                split_mask |= (mask_t)1 << i;
                                if ((mask >> i) & 0b1) {
                                    index++;
                                }
                            }
                            // Use adapted mask instead to completely fill data into one register
                            reg_t vec = tsl::loadu< ps >(myState.data_ptr);
                            vec = tsl::maskz_mov< ps >(mask, tsl::mask_binary_and(mask, split_mask));
                            tsl::storeu< ps >(data, vec);

                            // Replace mask with inverse of adapted mask to get remaining elements and adapt dataptr
                            mask_t new_mask = tsl::mask_binary_and(mask, tsl::mask_binary_not(split_mask))
                            *myState.bitlist_ptr = (mask_t)(new_mask >> index);
                            myState.data_ptr += index;
                        }
                        // Valid data fits into register with mask
                        else{
                            reg_t vec = tsl::loadu< ps >(myState.data_ptr);
                            vec = tsl::maskz_mov< ps >(mask, vec);
                            tsl::storeu< ps >(data, vec);

                            myState.bitlist_ptr++;
                            myState.data_ptr += element_count;
                        }
                    }
                    reg_t vec = tsl::loadu< ps >(start);
                    reg_t result = Operator< ps, tsl::workaround>::apply(vec);
                    tsl::storeu< ps >(myState.result_ptr, result);
                    myState.result_ptr += element_count;
                }
            }
        };

        static void flush(State& myState){
            const base_t *end = myState.data_ptr + myState.count;
            reg_t vec;
            while(myState.data_ptr <= end){
                *myState.result_ptr++ = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.data_ptr++);
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_CALC_UNARY_HPP