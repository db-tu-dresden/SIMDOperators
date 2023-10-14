#include "../src/SIMDOperators/operators/calc_unary.hpp"
#include <random>

template< typename ps , template <typename ...> typename Operator, typename StateType>
bool test_calc_unary(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;
    using mask_t = typename ps::imask_type;
    using StateDefault = typename tuddbs::calc_unary<ps, Operator>::State;
    using StateUnpacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist;
    using StatePacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist_packed;
    using StatePoslist = typename tuddbs::calc_unary<ps, Operator>::State_position_list;

    using side_t = typename std::conditional<std::is_same_v<StateType, StatePoslist>, typename ps::offset_base_type, typename ps::imask_type>::type;

    size_t side_count;
    std::unique_ptr<std::uniform_int_distribution<side_t>> dist_side;
    if constexpr(std::is_same_v<StateType, StateUnpacked>){
        side_count = 1 + ((data_count - 1) / ps::vector_element_count());
        dist_side = std::make_unique<std::uniform_int_distribution<side_t>>(0, (1ULL << ps::vector_element_count()) - 1);
    }else if(std::is_same_v<StateType, StatePacked>){
        side_count = 1 + ((data_count - 1) / (sizeof(mask_t)*8));
        dist_side = std::make_unique<std::uniform_int_distribution<side_t>>(0, std::numeric_limits<mask_t>::max());
    }else{
        side_count = data_count;
        dist_side = std::make_unique<std::uniform_int_distribution<side_t>>(0, data_count-1);
    }

    base_t* vec_ref = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* vec_test = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* test_result = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* ref_result = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    side_t* side_ref = reinterpret_cast<side_t*>(std::malloc(side_count*sizeof(side_t)));
    side_t* side_test = reinterpret_cast<side_t*>(std::malloc(side_count*sizeof(side_t)));

    memset(test_result, 0, data_count*sizeof(base_t));
    memset(ref_result, 0, data_count*sizeof(base_t));

    // Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());

    // Fill memory with test data
    base_t* temp_vec = vec_ref;
    base_t* temp_ref = ref_result;
    for(int i = 0; i < data_count; i++){
        *temp_vec++ = dist(rng);
    }
    std::memcpy(vec_test, vec_ref, data_count*sizeof(base_t));

    side_t* temp_side = side_ref;
    for(int i = 0; i < side_count; i++){
        *temp_side++ = (*dist_side)(rng);
    }
    std::memcpy(side_test, side_ref, side_count*sizeof(side_t));

    // Create States
    StateType state_ref, state_test;
    if constexpr(std::is_same_v<StateType, StateDefault>){
        state_ref = {.result_ptr = ref_result, .data_ptr = vec_ref, .count = data_count};
        state_test = {.result_ptr = test_result, .data_ptr = vec_test, .count = batch_size};
    }else if constexpr(std::is_same_v<StateType, StatePoslist>){
        state_ref = {.result_ptr = ref_result, .data_ptr = vec_ref, .pos_list_ptr = side_ref, .count = side_count};
        state_test = {.result_ptr = test_result, .data_ptr = vec_test, .pos_list_ptr = side_test, .count = batch_size};
    }else{
        state_ref = {.result_ptr = ref_result, .data_ptr = vec_ref, .bitlist_ptr = side_ref, .count = data_count};
        state_test = {.result_ptr = test_result, .data_ptr = vec_test, .bitlist_ptr = side_test, .count = batch_size};
    }

    // Testing
    tuddbs::calc_unary<ps, Operator>::flush(state_ref);
    
    if constexpr(!(std::is_same_v<StateType, StatePoslist>)){
        while((state_test.data_ptr - vec_test + ps::vector_element_count()) < data_count){
            tuddbs::calc_unary<ps, Operator>{}(state_test);
            const size_t t = vec_test + data_count - state_test.data_ptr;
            state_test.count = std::min(batch_size, t);
        }
        state_test.count = vec_test + data_count - state_test.data_ptr;
    }else{
        while((state_test.pos_list_ptr - side_test + ps::vector_element_count()) < data_count){
            tuddbs::calc_unary<ps, Operator>{}(state_test);
            const size_t t = side_test + data_count - state_test.pos_list_ptr;
            state_test.count = std::min(batch_size, t);
        }
        state_test.count = side_test + data_count - state_test.pos_list_ptr;
    }
    tuddbs::calc_unary<ps, Operator>::flush(state_test);

    //Check if Test is correct
    bool allOk = true;
    temp_vec = test_result;
    temp_ref = ref_result;
    while(temp_ref < (ref_result + data_count)){
        allOk &= (*temp_ref++ == *temp_vec++);
    }

    std::free(vec_ref);
    std::free(vec_test);
    std::free(test_result);
    std::free(ref_result);
    std::free(side_ref);
    std::free(side_test);
    return allOk;
}

template<typename ps, template <typename ...> typename Operator>
bool test_calc_unary_wrapper(const size_t batch_size, const size_t data_count){
    using StateDefault = typename tuddbs::calc_unary<ps, Operator>::State;
    using StateUnpacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist;
    using StatePacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist_packed;
    using StatePoslist = typename tuddbs::calc_unary<ps, Operator>::State_position_list;

    bool allOk = true;

    allOk &= test_calc_unary<ps, Operator, StateDefault>(batch_size, data_count);
    allOk &= test_calc_unary<ps, Operator, StateUnpacked>(batch_size, data_count);
    allOk &= test_calc_unary<ps, Operator, StatePacked>(batch_size, data_count);
    allOk &= test_calc_unary<ps, Operator, StatePoslist>(batch_size, data_count);

    return allOk;
}

int main()
{
    
    const int count = 16;
    bool allOk = true;
    std::cout << "Testing calc_unary...\n";
    std::cout.flush();
    // // INTEL - AVX512
    // {
    //     using ps_fit = typename tsl::simd<uint64_t, tsl::avx512>;
    //     using ps_no_fit = typename tsl::simd<uint32_t, tsl::avx512>;
        
    //     allOk &= test_calc_unary_wrapper<ps_fit, tsl::functors::inv>(ps_fit::vector_element_count(), count);
    //     allOk &= test_calc_unary_wrapper<ps_no_fit, tsl::functors::inv>(ps_no_fit::vector_element_count(), count);
    //     std::cout << "AVX512 Result: " << allOk << std::endl;
    // }
    // // INTEL - AVX2
    // {
    //     using ps_fit = typename tsl::simd<uint32_t, tsl::avx2>;
    //     using ps_no_fit = typename tsl::simd<uint64_t, tsl::avx2>;
        
    //     allOk &= test_calc_unary_wrapper<ps_fit, tsl::functors::inv>(ps_fit::vector_element_count(), count);
    //     allOk &= test_calc_unary_wrapper<ps_no_fit, tsl::functors::inv>(ps_no_fit::vector_element_count(), count);
    //     std::cout << "AVX2 Result: " << allOk << std::endl;
    // }
    // INTEL - SSE
    {
        using ps_fit = typename tsl::simd<uint16_t, tsl::sse>;
        using ps_no_fit = typename tsl::simd<uint64_t, tsl::sse>;

        allOk &= test_calc_unary_wrapper<ps_fit, tsl::functors::inv>(ps_fit::vector_element_count(), count);
        allOk &= test_calc_unary_wrapper<ps_no_fit, tsl::functors::inv>(ps_no_fit::vector_element_count(), count);
        std::cout << "SSE Result: " << allOk << std::endl;
    }
    // INTEL - SCALAR
    {
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;

        allOk &= test_calc_unary_wrapper<ps, tsl::functors::inv>(ps::vector_element_count(), count);
        std::cout << "Scalar Result: " << allOk << std::endl;
    }
    
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}
