#include "../src/SIMDOperators/operators/average.hpp"
#include <random>
#include <iostream>
#include <cmath>

template< typename ps, typename result_t = float>
bool test_average(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;

    base_t* vec_ref = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* vec_test = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));

    // Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(0, std::numeric_limits<base_t>::max());

    // Fill memory with test data
    base_t* temp_vec = vec_ref;
    for(int i = 0; i < data_count; i++){
        *temp_vec++ = dist(rng);
    }
    std::memcpy(vec_test, vec_ref, data_count*sizeof(base_t));

    // Create State
    typename tuddbs::average<ps, result_t>::State state_test = {.data_ptr = vec_test, .count = data_count};

    // Testing
    base_t divident = 0;
    for(int i = 0; i < data_count; i++){
        divident += vec_ref[i];
    }
    result_t result = (result_t)divident / (result_t)data_count;

    tuddbs::average<ps, result_t>{}(state_test);

    std::free(vec_ref);
    std::free(vec_test);
    return (std::abs(result - state_test.result) < 1e-6);
}

template<typename ps>
bool test_average_wrapper(const size_t batch_size, const size_t data_count){
    bool allOk = true;
    allOk &= test_average<ps>(batch_size, data_count);
    allOk &= test_average<ps, double>(batch_size, data_count);
    return allOk;
}

int main()
{
    const int count = 100000;
    bool allOk = true;
    std::cout << "Testing average...\n";
    std::cout.flush();
    // INTEL - AVX512
    {
        using ps_1 = typename tsl::simd<uint64_t, tsl::avx512>;
        using ps_2 = typename tsl::simd<uint32_t, tsl::avx512>;
        bool avx512_result = test_average_wrapper<ps_1>(ps_1::vector_element_count(), count);
        avx512_result &= test_average_wrapper<ps_2>(ps_2::vector_element_count(), count);
        allOk &= avx512_result;
        std::cout << "AVX512 Result: " << avx512_result << "\n------------------------------------------------" << std::endl;
    }
    // INTEL - AVX2
    {
        using ps_1 = typename tsl::simd<uint32_t, tsl::avx2>;
        using ps_2 = typename tsl::simd<uint64_t, tsl::avx2>;
        bool avx2_result = test_average_wrapper<ps_1>(ps_1::vector_element_count(), count);
        avx2_result &= test_average_wrapper<ps_2>(ps_2::vector_element_count(), count);
        allOk &= avx2_result;
        std::cout << "AVX2 Result: " << avx2_result << "\n------------------------------------------------" << std::endl;
    }
    // INTEL - SSE
    {
        using ps_1 = typename tsl::simd<uint16_t, tsl::sse>;
        using ps_2 = typename tsl::simd<uint64_t, tsl::sse>;
        bool sse_result = test_average_wrapper<ps_1>(ps_1::vector_element_count(), count);
        sse_result &= test_average_wrapper<ps_2>(ps_2::vector_element_count(), count);
        allOk &= sse_result;
        std::cout << "SSE Result: " << sse_result << "\n------------------------------------------------" << std::endl;
    }
    // INTEL - SCALAR
    {
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
        bool scalar_result = test_average_wrapper<ps>(ps::vector_element_count(), count);
        allOk &= scalar_result;
        std::cout << "Scalar Result: " << scalar_result << "\n------------------------------------------------" << std::endl;
    }
    
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}