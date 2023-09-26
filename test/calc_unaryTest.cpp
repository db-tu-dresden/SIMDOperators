#include "../src/SIMDOperators/operators/calc_unary.hpp"
#include <random>

template< typename ps , template <typename ...> typename Operator>
bool test_calc_unary(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;
    using reg_t = typename ps::register_type;
    using mask_t = typename ps::imask_type;
    using offset_t = typename ps::offset_base_type;

    size_t mask_count = 1 + ((data_count - 1) / ps::vector_element_count());

    base_t* vec_ref = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* vec_test = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    mask_t* mask_ref = reinterpret_cast<mask_t*>(std::malloc(mask_count*sizeof(mask_t)));
    mask_t* mask_test = reinterpret_cast<mask_t*>(std::malloc(mask_count*sizeof(mask_t)));
    base_t* test_result = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));
    base_t* ref_result = reinterpret_cast<base_t*>(std::malloc(data_count*sizeof(base_t)));

    memset(test_result, 0, data_count*sizeof(base_t));
    memset(ref_result, 0, data_count*sizeof(base_t));

    // Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());
    std::uniform_int_distribution<base_t> dist_mask(0, (1ULL << ps::vector_element_count()) - 1);

    // Fill memory with test data
    base_t* temp_vec = vec_ref;
    base_t* temp_ref = ref_result;
    for(int i = 0; i < data_count; i++){
        *temp_vec++ = dist(rng);
    }
    std::memcpy(vec_test, vec_ref, data_count*sizeof(base_t));

    mask_t* temp_mask = mask_ref;
    for(int i = 0; i < mask_count; i++){
        *temp_mask++ = dist_mask(rng);
    }
    std::memcpy(mask_test, mask_ref, mask_count*sizeof(mask_t));
    
    // Ref Value - TODO: use different implementation ?
    typename tuddbs::calc_unary<ps, Operator>::State_bitlist state_ref = {.result_ptr = ref_result, .data_ptr = vec_ref,.bitlist_ptr = mask_ref, .count = data_count};
    tuddbs::calc_unary<ps, Operator>::flush(state_ref);

    // Begin test
    typename tuddbs::calc_unary<ps, Operator>::State_bitlist state = {.result_ptr = test_result, .data_ptr = vec_test,.bitlist_ptr = mask_test, .count = batch_size};
    while((state.data_ptr - vec_test + ps::vector_element_count()) < data_count){
        tuddbs::calc_unary<ps, Operator>{}(state);

        const size_t t = vec_test + data_count - state.data_ptr;

        state.count = std::min(batch_size, t);
    }
    state.count = vec_test + data_count - state.data_ptr;
    tuddbs::calc_unary<ps, Operator>::flush(state);

    //Check if Test is correct
    bool allOk = true;
    temp_vec = test_result;
    temp_ref = ref_result;
    while(temp_ref < (ref_result + data_count)){
        allOk &= (*temp_ref++ == *temp_vec++);
    }

    
    std::free(test_result);
    std::free(ref_result);
    std::free(vec_ref);
    std::free(vec_test);
    std::free(mask_ref);
    std::free(mask_test);
    return allOk;
}

int main()
{
    const int count = 10000;
    bool allOk = true;
    std::cout << "Testing calc_unary...\n";
    std::cout.flush();
    // INTEL - AVX2
    {
        using ps = typename tsl::simd<uint64_t, tsl::avx2>;
        const size_t batch_size = 2 * ps::vector_element_count();

        allOk &= test_calc_unary<ps, tsl::functors::inv>(batch_size, count);
    }
    // INTEL - SSE
    {
        using ps = typename tsl::simd<uint16_t, tsl::sse>;
        const size_t batch_size = ps::vector_element_count();

        allOk &= test_calc_unary<ps, tsl::functors::inv>(batch_size, count);
    }
    // INTEL - SCALAR
    {
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
        const size_t batch_size = ps::vector_element_count();

        allOk &= test_calc_unary<ps, tsl::functors::inv>(batch_size, count);
    }
    
    std::cout << "Result: " << allOk << std::endl;
    return 0;
}
