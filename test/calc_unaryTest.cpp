#include "../src/SIMDOperators/operators/calc_unary.hpp"
#include <random>
#include <iostream>

template< typename ps , template <typename ...> typename Operator, typename StateType>
bool test_calc_unary(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;
    using mask_t = typename ps::imask_type;
    using StateDefault = typename tuddbs::calc_unary<ps, Operator>::State;
    using StateUnpacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist;
    using StatePacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist_packed;
    using StatePoslist = typename tuddbs::calc_unary<ps, Operator>::State_position_list;

    // "side" is refering to the needed side_data. E.g. bitlist_ptr or offset_ptr.
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

    // Fill memory with test data
    base_t* temp_vec = vec_ref;
    base_t* temp_ref = ref_result;
    for(int i = 0; i < data_count; i++){
        if constexpr(std::is_integral_v<base_t>){
            std::uniform_int_distribution<base_t> dist(0, std::numeric_limits<base_t>::max());
            *temp_vec++ = dist(rng);
        }else{
            std::uniform_real_distribution<base_t> dist(0, std::numeric_limits<base_t>::max());
            *temp_vec++ = dist(rng);
        }
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
bool do_test(const size_t data_count){
    std::cout << tsl::type_name<typename ps::base_type>() << " : [";
    using StateDefault = typename tuddbs::calc_unary<ps, Operator>::State;
    using StateUnpacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist;
    using StatePacked = typename tuddbs::calc_unary<ps, Operator>::State_bitlist_packed;
    using StatePoslist = typename tuddbs::calc_unary<ps, Operator>::State_position_list;
    
    bool allOk = true;
    bool temp = true;
    temp = test_calc_unary<ps, Operator, StateDefault>(ps::vector_element_count(), data_count);
    allOk &= temp;
    std::cout << temp << ", ";
    temp = test_calc_unary<ps, Operator, StateUnpacked>(ps::vector_element_count(), data_count);
    allOk &= temp;
    std::cout << temp << ", ";
    temp = test_calc_unary<ps, Operator, StatePacked>(ps::vector_element_count(), data_count);
    allOk &= temp;
    std::cout << temp << ", ";
    temp = test_calc_unary<ps, Operator, StatePoslist>(ps::vector_element_count(), data_count);
    allOk &= temp;
    std::cout << temp;
    std::cout << "] : " << allOk << std::endl;
    return allOk;
}

template<typename tsl_simd, template <typename ...> typename Operator>
bool test_calc_unary_wrapper(const size_t data_count){
    bool allOk = true;
    std::cout << "Using: " << tsl::type_name<tsl_simd>() << " with [Default, Unpacked, Packed, PosList]" <<std::endl;
    allOk &= do_test<tsl::simd<uint8_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<int8_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<uint16_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<int16_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<uint32_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<int32_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<uint64_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<int64_t, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<float, tsl_simd>, Operator>(data_count);
    allOk &= do_test<tsl::simd<double, tsl_simd>, Operator>(data_count);
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    return allOk;
}

int main()
{
    const int count = 1000000;
    bool allOk = true;
    std::cout << "Testing calc_unary...\n";
    std::cout.flush();
    allOk &= test_calc_unary_wrapper<tsl::scalar, tsl::functors::inv>(count);
    allOk &= test_calc_unary_wrapper<tsl::sse, tsl::functors::inv>(count);
    allOk &= test_calc_unary_wrapper<tsl::avx2, tsl::functors::inv>(count);
    allOk &= test_calc_unary_wrapper<tsl::avx512, tsl::functors::inv>(count);
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}
