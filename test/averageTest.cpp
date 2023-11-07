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

    // Fill memory with test data
    base_t* temp_vec = vec_ref;
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

    // Testing
    base_t divident = 0;
    for(int i = 0; i < data_count; i++){
        divident += vec_ref[i];
    }
    result_t result = (result_t) divident / (result_t) data_count;

    // Create State
    typename tuddbs::average<ps, result_t>::State state_test = {.data_ptr = vec_test, .count = batch_size};
    while((state_test.data_ptr - vec_test + ps::vector_element_count()) < data_count){
        tuddbs::average<ps, result_t>{}(state_test);
        const size_t t = vec_test + data_count - state_test.data_ptr;
        state_test.count = std::min(batch_size, t);
    }
    state_test.count = vec_test + data_count - state_test.data_ptr;
    tuddbs::average<ps, result_t>::finalize(state_test);

    std::free(vec_ref);
    std::free(vec_test);
    
    // isinf to catch to high results for e.g. float as base_type.
    if(std::isinf(result) && std::isinf(state_test.result)){
        return true;
    }else{
        return std::abs(result - state_test.result) < 1e-6;
    } 
}
template<typename ps>
bool do_test(const size_t data_count){
    bool allOk = true;
    allOk &= test_average<ps>(ps::vector_element_count(), data_count);
    allOk &= test_average<ps, double>(ps::vector_element_count(), data_count);
    allOk &= test_average<ps, float>(ps::vector_element_count(), data_count);
    std::cout << tsl::type_name<typename ps::base_type>() << " : " << allOk << std::endl;
    return allOk;
}

template<typename tsl_simd>
bool test_average_wrapper(const size_t data_count){
    bool allOk = true;
    std::cout << "Using: " << tsl::type_name<tsl_simd>() << std::endl;
    allOk &= do_test<tsl::simd<uint64_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<uint32_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<uint16_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<uint8_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<int64_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<int32_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<int16_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<int8_t, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<float, tsl_simd>>(data_count);
    allOk &= do_test<tsl::simd<double, tsl_simd>>(data_count);
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    return allOk;
}

int main()
{
    const int count = 100000;
    bool allOk = true;
    std::cout << "Testing average:\n";
    std::cout.flush();
    allOk &= test_average_wrapper<tsl::scalar>(count);
    allOk &= test_average_wrapper<tsl::sse>(count);
    allOk &= test_average_wrapper<tsl::avx2>(count);
    allOk &= test_average_wrapper<tsl::avx512>(count);
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}