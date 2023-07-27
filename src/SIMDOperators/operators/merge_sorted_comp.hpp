#include <cassert>

template < typename ps >
class merge_sorted_no_simd{
    using base_t = typename ps::base_type;

    public:
    struct State{
        base_t* result_ptr;
        base_t const* p_Data1Ptr;
        size_t p_CountData1;
        base_t const* p_Data2Ptr;
        size_t p_CountData2;
    };

    void operator()(State& myState){
        const base_t *endInPosL = myState.p_Data1Ptr + myState.p_CountData1;
        const base_t *endInPosR = myState.p_Data2Ptr + myState.p_CountData2;

        while (myState.p_Data1Ptr < endInPosL && myState.p_Data2Ptr < endInPosR){
            if(*myState.p_Data1Ptr < *myState.p_Data2Ptr){
                *myState.result_ptr = *myState.p_Data1Ptr;
                myState.result_ptr ++;
                myState.p_Data1Ptr ++; 
            }else if(*myState.p_Data1Ptr > *myState.p_Data2Ptr){
                *myState.result_ptr = *myState.p_Data2Ptr;
                myState.result_ptr ++;
                myState.p_Data2Ptr ++; 
            }else{
                *myState.result_ptr = *myState.p_Data2Ptr;
                myState.result_ptr ++;
                myState.p_Data2Ptr ++; 
                myState.p_Data1Ptr ++; 
            }
        }
    };

    static void flush(State& myState){
        const base_t *endInPosL = myState.p_Data1Ptr + myState.p_CountData1;
        const base_t *endInPosR = myState.p_Data2Ptr + myState.p_CountData2;

        while (myState.p_Data1Ptr < endInPosL){
            *myState.result_ptr = *myState.p_Data1Ptr;
            myState.p_Data1Ptr++;
            myState.result_ptr++;
        }

        while (myState.p_Data2Ptr < endInPosR){
            *myState.result_ptr = *myState.p_Data2Ptr;
            myState.p_Data2Ptr++;
            myState.result_ptr++;
        }
    };
};
