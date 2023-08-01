#ifndef SRC_OPERATORS_INTERSECT_SORTED_HPP
#define SRC_OPERATORS_INTERSECT_SORTED_HPP

#include "/home/tucholke/work/TSLGen/generated_tsl/generator_output/include/tslintrin.hpp"  //TODO: adapt include path to tslgen
#include <cassert>

namespace tuddbs {
    
    template< typename ps >
    class intersect_sorted{
        using base_t = typename ps::base_type;
        using mask_t = typename ps::imask_type;
        using reg_t = typename ps::register_type;

        public:
        struct State{
            base_t* result_ptr;
            base_t const* p_Data1Ptr;
            size_t p_CountData1;
            base_t const* p_Data2Ptr;
            size_t p_CountData2;
        };

        void operator()(State& myState){
            size_t element_count = ps::vector_element_count();

            const base_t *pl = myState.p_Data1Ptr;
            const base_t *pr = myState.p_Data2Ptr;

            const base_t *endInPosL = myState.p_Data1Ptr + myState.p_CountData1;
            const base_t *endInPosR = myState.p_Data2Ptr + myState.p_CountData2;

            reg_t data1Vector = tsl::set1< ps >(*myState.p_Data1Ptr);
            reg_t data2Vector = tsl::loadu< ps >(myState.p_Data2Ptr);
            mask_t full_hit = tsl::to_integral<ps>(tsl::equal< ps >(data1Vector, data1Vector));

            // While data can fit into simd-register compare data by broadcasting one value of the first data array and comparing it with the second array.
            while (myState.p_Data2Ptr <= (endInPosR - element_count) && myState.p_Data1Ptr <= (endInPosL - element_count)) {
                mask_t resultMaskEqual = tsl::to_integral<ps>(tsl::equal< ps >(data2Vector, data1Vector));
                mask_t maskLess = tsl::to_integral<ps>(tsl::less_than< ps >(data2Vector, data1Vector));

                if(resultMaskEqual != 0){
                    *myState.result_ptr = *myState.p_Data1Ptr;
                    myState.result_ptr++;
                }
                if(maskLess == 0){
                    myState.p_Data1Ptr++;
                    data1Vector = tsl::set1< ps >(*myState.p_Data1Ptr);
                }else{
                    if(maskLess == full_hit){
                        myState.p_Data2Ptr += element_count;
                        data2Vector = tsl::loadu< ps >(myState.p_Data2Ptr);
                    }else{
                        myState.p_Data1Ptr++;
                        data1Vector = tsl::set1< ps >(*myState.p_Data1Ptr);
                        myState.p_Data2Ptr += tsl::mask_population_count< ps >(maskLess);
                        data2Vector = tsl::loadu< ps >(myState.p_Data2Ptr);
                    }
                }
            }
        };

        static void flush(State& myState){
            const base_t *endInPosL = myState.p_Data1Ptr + myState.p_CountData1;
            const base_t *endInPosR = myState.p_Data2Ptr + myState.p_CountData2;
            
            // Use this code if all values are unique
            while (myState.p_Data1Ptr < endInPosL && myState.p_Data2Ptr < endInPosR) {
                if (*myState.p_Data1Ptr < *myState.p_Data2Ptr) {
                    ++myState.p_Data1Ptr;
                } else if (*myState.p_Data2Ptr < *myState.p_Data1Ptr) {
                    ++myState.p_Data2Ptr;
                } else {
                    *myState.result_ptr++ = *myState.p_Data1Ptr++;
                    ++myState.p_Data2Ptr;
                }
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_INTERSECT_SORTED_HPP