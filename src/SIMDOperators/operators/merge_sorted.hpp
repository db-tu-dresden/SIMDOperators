#ifndef SRC_OPERATORS_MERGE_HPP
#define SRC_OPERATORS_MERGE_HPP

#include "/home/tucholke/work/TSLGen/generated_tsl/generator_output/include/tslintrin.hpp"
#include <cassert>

namespace tuddbs {
    
    template< typename ps >
    class merge_sorted{
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

            // While data can fit into simd-register compare data by broadcasting one value of the first data array and comparing it with the second array.
            while (myState.p_Data2Ptr <= (endInPosR - element_count) && myState.p_Data1Ptr <= (endInPosL - element_count)) {
                mask_t resultMaskEqual = tsl::to_integral<ps>(tsl::equal< ps >(data2Vector, data1Vector));
                mask_t m_MaskGreater = tsl::to_integral<ps>(tsl::greater_than< ps >(data1Vector, data2Vector));

                // If value of v1 is the smallest, store it into result and broadcast next value.
                if ((m_MaskGreater) == 0) {
                    if (resultMaskEqual == 0) {
                        *myState.result_ptr = tsl::extract_value< ps, 0 >(data1Vector);
                        myState.result_ptr++;
                    }
                    myState.p_Data1Ptr++;
                    data1Vector = tsl::set1< ps >(*myState.p_Data1Ptr);
                } 
                // Else store all values that are greater than v1 into result.
                else {
                    tsl::compress_store< ps >(m_MaskGreater, myState.result_ptr, data2Vector);
                    myState.p_Data2Ptr += tsl::mask_population_count< ps >(m_MaskGreater);
                    myState.result_ptr += tsl::mask_population_count< ps >(m_MaskGreater);
                    data2Vector = tsl::loadu< ps >(myState.p_Data2Ptr);
                }
            }
        };

        static void flush(State& myState){
            const base_t *endInPosL = myState.p_Data1Ptr + myState.p_CountData1;
            const base_t *endInPosR = myState.p_Data2Ptr + myState.p_CountData2;

            // size_t element_count = ps::vector_element_count();
            // reg_t data1Vector, data2Vector;

            // if(myState.p_CountData1 >= element_count){
            //     data1Vector = tsl::loadu<ps>(myState.p_Data1Ptr);
            // }
            // reg_t data1Vector = tsl::set1< ps >(*myState.p_Data1Ptr);
            // reg_t data2Vector = tsl::loadu< ps >(myState.p_Data2Ptr);


            // while (myState.p_Data1Ptr < endInPosL && myState.p_Data2Ptr < endInPosR) {
            //     if (*myState.p_Data1Ptr < *myState.p_Data2Ptr) {
            //         cout << "c1(" << *myState.p_Data1Ptr << "), ";
            //         *myState.result_ptr = *myState.p_Data1Ptr;
            //         myState.p_Data1Ptr++;
            //     } else if (*myState.p_Data2Ptr < *myState.p_Data1Ptr) {
            //         cout << "c2(" << *myState.p_Data2Ptr << "), ";
            //         *myState.result_ptr = *myState.p_Data2Ptr;
            //         myState.p_Data2Ptr++;
            //     } else {// *inPosL == *inPosR
            //         cout << "c3(" << *myState.p_Data1Ptr << "), ";
            //         *myState.result_ptr = *myState.p_Data1Ptr;
            //         myState.p_Data1Ptr++;
            //         myState.p_Data2Ptr++;
            //     }
            //     myState.result_ptr++;
            // }

            // while (myState.p_Data1Ptr < (endInPosL - element_count)) {
            //     cout << "l1(" << *myState.p_Data1Ptr << "), ";
            //     data1Vector = tsl::loadu< ps >(myState.p_Data1Ptr);
            //     tsl::storeu< ps >(myState.result_ptr, data1Vector);
            //     myState.result_ptr += element_count;
            //     myState.p_Data1Ptr += element_count;
            // }

            // while (myState.p_Data2Ptr < (endInPosR - element_count)) {
            //     cout << "l2(" << *myState.p_Data2Ptr<<"), ";
            //     data2Vector = tsl::loadu< ps >(myState.p_Data2Ptr);
            //     tsl::storeu< ps >(myState.result_ptr, data2Vector);
            //     myState.p_Data2Ptr += element_count;
            //     myState.result_ptr += element_count;
            // }
            // cout << "Flush:[";
            
            // Traverse both arrays and store the smaller element in result
            while(myState.p_Data1Ptr < endInPosL && myState.p_Data2Ptr < endInPosR){
                if (*myState.p_Data1Ptr < *myState.p_Data2Ptr) {
                    *myState.result_ptr = *myState.p_Data1Ptr;
                    myState.p_Data1Ptr++;
                } else if (*myState.p_Data2Ptr < *myState.p_Data1Ptr) {
                    *myState.result_ptr = *myState.p_Data2Ptr;
                    myState.p_Data2Ptr++;
                } else {// *inPosL == *inPosR
                    *myState.result_ptr = *myState.p_Data1Ptr;
                    myState.p_Data1Ptr++;
                    myState.p_Data2Ptr++;
                }
                myState.result_ptr++;
            }
            
            //Store remaining elements of first data pointer
            while (myState.p_Data1Ptr < endInPosL) {
                *myState.result_ptr = *myState.p_Data1Ptr;
                myState.p_Data1Ptr ++;
                myState.result_ptr ++;
            }
            
            //Store remaining elements of second data pointer
            while (myState.p_Data2Ptr < endInPosR) {
                *myState.result_ptr = *myState.p_Data2Ptr;
                myState.p_Data2Ptr ++;
                myState.result_ptr ++;
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_MERGE_SORTED_HPP