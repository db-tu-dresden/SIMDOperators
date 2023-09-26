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
#include <bitset>

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
            mask_t* bitlist_ptr;
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
            State_bitlist backup = myState;
            size_t const element_count = ps::vector_element_count();
            size_t const mask_size = sizeof(mask_t) * 8;
            const int mask_count = 1 + ((myState.count - 1) / (sizeof(mask_t)*8));

            const base_t *end = myState.data_ptr + myState.count;
            const mask_t *mask_end = myState.bitlist_ptr + mask_count;

            if constexpr(element_count < mask_size){
                // darauf achten, dass zuerst die relevanten bits verwendet werden (Stichwort: packed )
            }else{
                base_t* data = reinterpret_cast<base_t*>(std::malloc(element_count*sizeof(base_t)));
                base_t* start = data;
                while(myState.data_ptr <= (end - element_count)){
                    data = start;
                    int mask_pop = 0;
                    while(mask_pop < element_count){
                        mask_t mask = *myState.bitlist_ptr;
                        int old_count = mask_pop;
                        mask_pop += tsl::mask_population_count< ps >(mask);
                        std::cout << "mask_pop: " << mask_pop << std::endl;
                        // If all masks combined can not fill one register, flush and update state
                        if(myState.bitlist_ptr > mask_end && mask_pop < element_count){
                            std::cout << "no load possible -> flushing" << std::endl;
                            flush(backup);
                            myState = backup;
                            return;
                        }
                        // mask_pop is greater than element count -> split mask so data fits again
                        if(mask_pop > element_count){
                            // Find index where to split the mask
                            size_t index = 0;
                            mask_t split_mask = 0;
                            for (size_t i = 0; index < mask_size && i < element_count - old_count; index++) {
                                split_mask |= (mask_t)1 << index;
                                if ((mask >> index) & 0b1) {
                                    i++;
                                }
                            }
                            // Use adapted mask instead to completely fill data into one register
                            mask_t new_mask = tsl::mask_binary_and< ps >(mask, split_mask);
                            reg_t vec_temp = tsl::loadu< ps >(myState.data_ptr);

                            // std::cout << "data: [";
                            // for(int i = 0; i < element_count; i++){
                            //     std::cout << start[i] << ", ";
                            // }
                            // std::cout << "]" << std::endl;

                            tsl::compress_store< ps >(new_mask, data, vec_temp);

                            std::bitset<8> b(mask);
                            std::bitset<8> i(new_mask);
                            
                            // Replace mask with inverse of adapted mask to get remaining elements and adapt dataptr
                            new_mask = tsl::mask_binary_and< ps >(mask, tsl::mask_binary_not< ps >(split_mask));
                            std::bitset<8> n(new_mask);
                            std::cout << "IF mask: " << b << ", mask_split: " << i << ", new_mask: " << n << std::endl;
                            *myState.bitlist_ptr = new_mask;
                        }
                        // Valid data fits into register with mask
                        else{
                            tsl::compress_store< ps >(mask, data, tsl::loadu< ps >(myState.data_ptr));
                            myState.bitlist_ptr++;
                            myState.data_ptr += element_count;

                            data += tsl::mask_population_count< ps >(mask);
                        }
                    }
                    reg_t vec = tsl::loadu< ps >(start);
                    reg_t result = Operator< ps, tsl::workaround>::apply(vec);
                    tsl::storeu< ps >(myState.result_ptr, result);
                    myState.result_ptr += element_count;
                    backup = myState;
                }
                std::free(start);
            }
        };

        static void flush(State& myState){
            const base_t *end = myState.data_ptr + myState.count;
            while(myState.data_ptr <= end){
                *myState.result_ptr++ = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.data_ptr++);
            }
        };

        static void flush(State_bitlist& myState){
            const int mask_count = 1 + ((myState.count - 1) / (sizeof(mask_t)*8));
            const mask_t* end = myState.bitlist_ptr + mask_count;

            while(myState.bitlist_ptr < end){
                mask_t mask = *myState.bitlist_ptr++;
                for(size_t i = 0; i < sizeof(mask_t)*8; i++){
                    if((mask >> i) & 0b1){
                        *myState.result_ptr++ = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.data_ptr);
                    }
                    myState.data_ptr++;
                }
            }
        }
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_CALC_UNARY_HPP