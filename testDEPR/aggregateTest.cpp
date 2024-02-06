#include "../src/SIMDOperators/operators/aggregate.hpp"
#include <random>
#include <iostream>
#include <cmath>

template< typename ps , template <typename ...> typename Op, template <typename ...> typename Op_h>
bool test_aggregate(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;
    using mask_t = typename ps::imask_type;

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

    // Create States
    size_t test_size;
    (data_count < ps::vector_element_count())? test_size = data_count : test_size = batch_size;
    typename tuddbs::aggregate<ps, Op, Op_h>::State state_ref(vec_ref, data_count);
    typename tuddbs::aggregate<ps, Op, Op_h>::State state_test(vec_test, test_size);

    // Testing
    tuddbs::aggregate<ps, Op, Op_h>::flush(state_ref);
    
    while((state_test.data_ptr - vec_test + ps::vector_element_count()) < data_count){
        tuddbs::aggregate<ps, Op, Op_h>{}(state_test);
        const size_t t = vec_test + data_count - state_test.data_ptr;
        state_test.count = std::min(batch_size, t);
    }
    state_test.count = vec_test + data_count - state_test.data_ptr;
    tuddbs::aggregate<ps, Op, Op_h>::flush(state_test);

    std::free(vec_ref);
    std::free(vec_test);
    bool allOk;
    if constexpr(std::is_integral_v<base_t>){
        allOk = (state_ref.result == state_test.result);
    }else{
        // if both are INF, they are equal.
        if(std::isinf(state_ref.result) && std::isinf(state_test.result)){
            return true;
        }
        allOk = (std::abs(state_ref.result - state_test.result) < 1e-6);
    }
    if(!allOk)std::cout << std::fixed << state_ref.result << " == " << state_test.result << std::endl;
    return allOk;
}

template<typename ps>
bool do_test(const size_t data_count){
    bool allOk = true;
    // Non integral datatypes tend to become NaN with OR-Aggregation. NaN cant be compared, so we leave them out.
    if constexpr(std::is_integral_v<typename ps::base_type>){
        allOk &= test_aggregate<ps, tsl::functors::binary_or, tsl::functors::hor>(ps::vector_element_count(), data_count);
    }
    allOk &= test_aggregate<ps, tsl::functors::add, tsl::functors::hadd>(ps::vector_element_count(), data_count);
    allOk &= test_aggregate<ps, tsl::functors::min, tsl::functors::hmin>(ps::vector_element_count(), data_count);
    allOk &= test_aggregate<ps, tsl::functors::max, tsl::functors::hmax>(ps::vector_element_count(), data_count);
    std::cout << tsl::type_name<typename ps::base_type>() << " : " << allOk << std::endl;
    return allOk;
}

template<typename tsl_simd>
bool test_aggregate_wrapper(const size_t data_count){
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
    std::cout << "Testing aggregate...\n";
    std::cout.flush();
    allOk &= test_aggregate_wrapper<tsl::scalar>(count);
    allOk &= test_aggregate_wrapper<tsl::sse>(count);
    allOk &= test_aggregate_wrapper<tsl::avx2>(count);
    allOk &= test_aggregate_wrapper<tsl::avx512>(count);
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}