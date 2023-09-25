#include "../src/SIMDOperators/operators/calc_unary.hpp"
#include <random>

template< typename ps , template <typename ...> typename Operator>
bool test_calc_unary(const size_t batch_size, const size_t data_count){
    using base_t = typename ps::base_type;
    using reg_t = typename ps::register_type;

    base_t* vec = reinterpret_cast<base_t*>(malloc(data_count*sizeof(base_t)));
    base_t* test_result = reinterpret_cast<base_t*>(malloc(data_count*sizeof(base_t)));
    base_t* ref_result = reinterpret_cast<base_t*>(malloc(data_count*sizeof(base_t)));

    base_t* temp_vec = vec;
    base_t* temp_ref = ref_result;

    memset(test_result, 0, data_count*sizeof(base_t));
    memset(ref_result, 0, data_count*sizeof(base_t));

    // Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());

    // Fill memory with test data
    for(int i = 0; i < data_count; i++){
        *temp_vec++ = dist(rng);
    }
    temp_vec = vec;

    // Ref Value
    while(temp_vec  <= (vec + data_count)){
        *temp_ref++ = Operator< tsl::simd<base_t, tsl::scalar>, tsl::workaround>::apply(*temp_vec++);
    }
    temp_vec = vec;
    temp_ref = ref_result;

    // Begin test
    typename tuddbs::calc_unary<ps, Operator>::State state = {.result_ptr = test_result, .data_ptr = vec, .count = data_count};
    while((state.data_ptr - vec + ps::vector_element_count()) < data_count){
        tuddbs::calc_unary<ps, Operator>{}(state);

        const size_t t = vec + data_count - state.data_ptr;

        state.count = std::min(batch_size, t);
    }
    state.count = vec + data_count - state.data_ptr;
    tuddbs::calc_unary<ps, Operator>::flush(state);

    //Check if Test is correct
    bool allOk = true;
    temp_vec = test_result;
    while(allOk && temp_ref <= (ref_result + data_count)){
        allOk &= (*temp_ref++ == *temp_vec++);
    }

    free(vec);
    free(test_result);
    free(ref_result);
    return allOk;
}

template<typename ps , template <typename ...> typename Operator, typename state_t>
state_t calc_unary_exec(state_t state, size_t batch_size){
    typename ps::base_type* vec = state.data_ptr;
    while((state.data_ptr - vec + ps::vector_element_count()) < state.count){
        tuddbs::calc_unary<ps, Operator>{}(state);

        const size_t t = vec + state.count - state.data_ptr;

        state.count = std::min(batch_size, t);
    }
    state.count = vec + state.count - state.data_ptr;
    tuddbs::calc_unary<ps, Operator>::flush(state);
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
        const size_t batch_size = ps::vector_element_count();

        allOk &= test_calc_unary<ps, tsl::functors::inv>(batch_size, count);
    }
    // INTEL - SSE
    {
        using ps = typename tsl::simd<uint64_t, tsl::sse>;
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
