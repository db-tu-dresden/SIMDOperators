#include "../src/SIMDOperators/operators/aggregate.hpp"
#include <random>

template< typename ps , template <typename ...> typename Operator>
bool test_aggregate(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;
    using mask_t = typename ps::imask_type;

    base_t* vec_ref = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* vec_test = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));

    // Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());

    // Fill memory with test data
    base_t* temp_vec = vec_ref;
    for(int i = 0; i < data_count; i++){
        *temp_vec++ = dist(rng);
    }
    std::memcpy(vec_test, vec_ref, data_count*sizeof(base_t));

    // Create States
    typename tuddbs::aggregate<ps, Operator>::State state_ref, state_test;
    state_ref = {.data_ptr = vec_ref, .count = data_count};
    state_test = {.data_ptr = vec_test, .count = batch_size};
   

    // Testing
    tuddbs::aggregate<ps, Operator>::flush(state_ref);
    
    while((state_test.data_ptr - vec_test + ps::vector_element_count()) < data_count){
        tuddbs::aggregate<ps, Operator>{}(state_test);
        const size_t t = vec_test + data_count - state_test.data_ptr;
        state_test.count = std::min(batch_size, t);
    }
    state_test.count = vec_test + data_count - state_test.data_ptr;

    tuddbs::aggregate<ps, Operator>::flush(state_test);

    std::free(vec_ref);
    std::free(vec_test);
    std::bitset<sizeof(base_t)*8> test_(state_test.result);
    std::bitset<sizeof(base_t)*8> ref_(state_ref.result);

    std::cout << "Ref: " << ref_ << "\tTest: " << test_ << std::endl;

    return (state_ref.result == state_test.result);
}

template<typename ps>
bool test_aggregate_wrapper(const size_t batch_size, const size_t data_count){
    bool allOk = true;

    allOk &= test_aggregate<ps, tsl::functors::hor>(batch_size, data_count);

    return allOk;
}

int main()
{
    
    const int count = 16;
    bool allOk = true;
    std::cout << "Testing aggregate...\n";
    std::cout.flush();
    // // INTEL - AVX512
    // {
    //     using ps_1 = typename tsl::simd<uint64_t, tsl::avx512>;
    //     using ps_2 = typename tsl::simd<uint32_t, tsl::avx512>;
        
    //     allOk &= test_aggregate_wrapper<ps_1>(ps_1::vector_element_count(), count);
    //     allOk &= test_aggregate_wrapper<ps_2>(ps_2::vector_element_count(), count);
    //     std::cout << "AVX512 Result: " << allOk << std::endl;
    // }
    // INTEL - AVX2
    {
        using ps_1 = typename tsl::simd<uint32_t, tsl::avx2>;
        using ps_2 = typename tsl::simd<uint64_t, tsl::avx2>;
        
        allOk &= test_aggregate_wrapper<ps_1>(ps_1::vector_element_count(), count);
        allOk &= test_aggregate_wrapper<ps_2>(ps_2::vector_element_count(), count);
        std::cout << "AVX2 Result: " << allOk << std::endl;
    }
    // INTEL - SSE
    {
        using ps_1 = typename tsl::simd<uint16_t, tsl::sse>;
        using ps_2 = typename tsl::simd<uint64_t, tsl::sse>;

        allOk &= test_aggregate_wrapper<ps_1>(ps_1::vector_element_count(), count);
        allOk &= test_aggregate_wrapper<ps_2>(ps_2::vector_element_count(), count);
        std::cout << "SSE Result: " << allOk << std::endl;
    }
    // INTEL - SCALAR
    {
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;

        allOk &= test_aggregate_wrapper<ps>(ps::vector_element_count(), count);
        std::cout << "Scalar Result: " << allOk << std::endl;
    }
    
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}