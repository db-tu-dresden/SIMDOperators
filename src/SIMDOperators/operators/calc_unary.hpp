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

#include <tslintrin.hpp>
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

        struct State_bitlist{
            base_t* result_ptr;
            base_t const* data_ptr;
            mask_t* bitlist_ptr;
            size_t count;
        };

        struct State_bitlist_packed{
            base_t* result_ptr;
            base_t const* data_ptr;
            mask_t* bitlist_ptr;
            size_t count;
        };

        struct State_position_list{
            base_t* result_ptr;
            base_t const* data_ptr;
            offset_t* pos_list_ptr;
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
            const base_t *end = myState.data_ptr + myState.count;
            // To catch wrong masks
            const mask_t shift = (1 << ps::vector_element_count()) - 1;

            while(myState.data_ptr <= (end - element_count)){
                typename ps::mask_type mask = tsl::to_mask< ps >(*myState.bitlist_ptr++ & shift);

                reg_t vec = tsl::loadu< ps >(myState.data_ptr);
                reg_t temp = Operator< ps, tsl::workaround>::apply(vec);
                reg_t result = tsl::blend< ps >(mask, vec, temp);

                tsl::storeu<ps>(myState.result_ptr, result);

                myState.data_ptr += element_count;
                myState.result_ptr += element_count;
            }
        };

        void operator()(State_bitlist_packed& myState){
            size_t const element_count = ps::vector_element_count();
            const base_t *end = myState.data_ptr + myState.count;
            const mask_t shift = (1 << ps::vector_element_count()) - 1;

            while(myState.data_ptr <= (end - element_count)){
                mask_t mask = *myState.bitlist_ptr++;
                for(int i = 0; i < sizeof(mask_t)*8; i += element_count){
                    typename ps::mask_type new_mask = tsl::to_mask< ps >((mask >> i) & shift);

                    reg_t vec = tsl::loadu< ps >(myState.data_ptr);
                    reg_t temp = Operator< ps, tsl::workaround>::apply(vec);                    
                    reg_t result = tsl::blend< ps >(new_mask, vec, temp);

                    tsl::storeu<ps>(myState.result_ptr, result);

                    myState.data_ptr += element_count;
                    myState.result_ptr += element_count;
                }
            }
        };

        void operator()(State_position_list& myState){
            using reg_offset = typename ps::offset_base_register_type;
            using ps_offset = typename tsl::simd<offset_t, typename ps::target_extension>;
            size_t const element_count = ps::vector_element_count();
            const offset_t* end = myState.pos_list_ptr + myState.count;

            while(myState.pos_list_ptr <= (end - element_count)){
                reg_offset pos_vec = tsl::loadu< ps_offset >(myState.pos_list_ptr);
                reg_t result = tsl::gather< ps, ps_offset >(myState.data_ptr, pos_vec);
                result = Operator< ps, tsl::workaround>::apply(result);
                tsl::storeu< ps >(myState.result_ptr, result);
                myState.result_ptr += element_count;
                myState.pos_list_ptr += element_count;
            }
        };

        static void flush(State& myState){
            const base_t *end = myState.data_ptr + myState.count;
            while(myState.data_ptr <= end){
                *myState.result_ptr++ = Operator<tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.data_ptr++);
            }
        };

        static void flush(State_bitlist& myState){
            const int mask_count = 1 + ((myState.count - 1) / ps::vector_element_count());
            const mask_t* end_mask = myState.bitlist_ptr + mask_count;
            // To catch wrong masks
            const mask_t shift = (1 << ps::vector_element_count()) - 1;

            while(myState.bitlist_ptr < end_mask){
                mask_t mask = *myState.bitlist_ptr++ & shift;
                for(size_t i = 0; i < ps::vector_element_count(); i++){
                    if((mask >> i) & 0b1){
                        *myState.result_ptr = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.data_ptr);
                    }else{
                        *myState.result_ptr = *myState.data_ptr;
                    }
                    myState.result_ptr++;
                    myState.data_ptr++;
                }
            }
        };

        static void flush(State_bitlist_packed& myState){
            const int mask_count = 1 + ((myState.count - 1) / (sizeof(mask_t)*8));
            const mask_t* end_mask = myState.bitlist_ptr + mask_count;

            while(myState.bitlist_ptr < end_mask){
                mask_t mask = *myState.bitlist_ptr++;
                for(size_t i = 0; i < sizeof(mask_t)*8; i++){
                    if((mask >> i) & 0b1){
                        *myState.result_ptr = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*myState.data_ptr);
                    }else{
                        *myState.result_ptr = *myState.data_ptr;
                    }
                    myState.result_ptr++;
                    myState.data_ptr++;
                }
            }
        };

        static void flush(State_position_list& myState){
            const offset_t* end = myState.pos_list_ptr + myState.count;

            while(myState.pos_list_ptr < end){
                base_t value = *(myState.data_ptr + *myState.pos_list_ptr++);
                *myState.result_ptr++ = Operator<tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(value);
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_CALC_UNARY_HPP

// Following code is a version with projection in it. Maybe interesting for the future.

// void operator()(State_bitlist_packed& myState){
//     size_t const element_count = ps::vector_element_count();
//     const int mask_count = 1 + ((myState.count - 1) / (sizeof(mask_t)*8));
//     const mask_t shift = (1 << ps::vector_element_count()) - 1;
//     const base_t *end = myState.data_ptr + myState.count;
//     const mask_t *mask_end = myState.bitlist_ptr + mask_count;

//     base_t* data = reinterpret_cast<base_t*>(std::malloc(element_count*sizeof(base_t)));
//     base_t* start = data;

//     // Processing of packed masks
//     while(myState.data_ptr <= (end - element_count)){
//         data = start;
//         int mask_pop = 0;

//         if(myState.used_bits){
//             while(myState.used_bits != sizeof(mask_t)*8){
//                 mask_t mask = *myState.bitlist_ptr;
//                 if(mask & 0b1){
//                     *data++ = *myState.data_ptr;
//                     mask_pop ++;
//                 }
//                 myState.data_ptr++;
//                 myState.used_bits++;
//                 *myState.bitlist_ptr = *myState.bitlist_ptr >> 1;
//                 if(mask_pop == element_count){
//                     break;
//                 }
//             }
//         }
//         while(mask_pop < element_count){
//             if(myState.used_bits == sizeof(mask_t)*8){
//                 myState.bitlist_ptr++;
//                 myState.used_bits = 0;
//             }
//             // If all masks combined can not fill one register -> flush
//             if(myState.bitlist_ptr >= mask_end){
//                 data = start;
//                 for(size_t i = 0; i < mask_pop; i++){
//                     *myState.result_ptr++ = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*data++);
//                 }
//                 return;
//             }
            
//             mask_t mask = *myState.bitlist_ptr;
//             int old_count = mask_pop;
//             mask_pop += tsl::mask_population_count< ps >(mask & shift);
            
//             // mask_pop is greater than element count -> split mask so data fits again
//             if(mask_pop > element_count){
//                 // Find index where to split the mask
//                 size_t index = 0;
//                 mask_t split_mask = 0;
//                 for (size_t i = 0; index < element_count && i < element_count - old_count; index++) {
//                     split_mask |= (mask_t)1 << index;
//                     if ((mask >> index) & 0b1) {
//                         i++;
//                     }
//                 }
//                 // Use adapted mask instead to completely fill data into one register
//                 mask_t new_mask = tsl::mask_binary_and< ps >(mask, split_mask);
//                 reg_t vec_temp = tsl::loadu< ps >(myState.data_ptr);
//                 tsl::compress_store< ps >(new_mask, data, vec_temp);

//                 // Replace mask with inverse of adapted mask to get remaining elements
//                 *myState.bitlist_ptr = (mask & ~split_mask) >> index;
//                 myState.used_bits += index;
//                 myState.data_ptr += index;
//             }
//             // Valid data fits into register with mask
//             else{
//                 tsl::compress_store< ps >(mask & shift, data, tsl::loadu< ps >(myState.data_ptr));
//                 data += tsl::mask_population_count< ps >(mask & shift);
//                 myState.data_ptr += element_count;
//                 myState.used_bits += element_count;
//                 *myState.bitlist_ptr = mask >> element_count;
//             }
            
//         }
//         reg_t vec = tsl::loadu< ps >(start);
//         reg_t result = Operator< ps, tsl::workaround>::apply(vec);
//         tsl::storeu< ps >(myState.result_ptr, result);
//         myState.result_ptr += element_count;
//     }
//     std::free(start);
// };


// void operator()(State_bitlist& myState){
//     State_bitlist backup = myState;
//     size_t const element_count = ps::vector_element_count();
//     const int mask_count = 1 + ((myState.count - 1) / (sizeof(mask_t)*8));
//     const mask_t shift = (1 << ps::vector_element_count()) - 1;
//     const base_t *end = myState.data_ptr + myState.count;
//     const mask_t *mask_end = myState.bitlist_ptr + mask_count;

//     base_t* data = reinterpret_cast<base_t*>(std::malloc(element_count*sizeof(base_t)));
//     base_t* start = data;

//     while(myState.data_ptr <= (end - element_count)){
//         data = start;
//         int mask_pop = 0;
//         while(mask_pop < element_count){
//             mask_t mask = *myState.bitlist_ptr & shift;
//             int old_count = mask_pop;
//             mask_pop += tsl::mask_population_count< ps >(mask);
//             // If all masks combined can not fill one register, flush and update state
//             if(myState.bitlist_ptr >= mask_end && mask_pop < element_count){
//                 flush(backup);
//                 myState = backup;
//                 return;
//             }
//             // mask_pop is greater than element count -> split mask so data fits again
//             else if(mask_pop > element_count){
//                 // Find index where to split the mask
//                 size_t index = 0;
//                 mask_t split_mask = 0;
//                 for (size_t i = 0, index = 0; index < element_count && i < element_count - old_count; index++) {
//                     split_mask |= (mask_t)1 << index;
//                     if ((mask >> index) & 0b1) {
//                         i++;
//                     }
//                 }
//                 // Use adapted mask instead to completely fill data into one register
//                 mask_t new_mask = tsl::mask_binary_and< ps >(mask, split_mask);
//                 reg_t vec_temp = tsl::loadu< ps >(myState.data_ptr);
//                 tsl::compress_store< ps >(new_mask, data, vec_temp);

//                 // Replace mask with inverse of adapted mask to get remaining elements
//                 new_mask = tsl::mask_binary_and< ps >(mask, tsl::mask_binary_not< ps >(split_mask));
//                 *myState.bitlist_ptr = new_mask;
//             }
//             // Valid data fits into register with mask
//             else{
//                 tsl::compress_store< ps >(mask, data, tsl::loadu< ps >(myState.data_ptr));
//                 myState.bitlist_ptr++;
//                 myState.data_ptr += element_count;

//                 data += tsl::mask_population_count< ps >(mask);
//             }
//         }
//         reg_t vec = tsl::loadu< ps >(start);
//         reg_t result = Operator< ps, tsl::workaround>::apply(vec);
//         tsl::storeu< ps >(myState.result_ptr, result);
//         myState.result_ptr += element_count;
//         backup = myState;
//     }
//     std::free(start);
// };