#include "../src/SIMDOperators/operators/merge_sorted.hpp"
#include <algorithm>
#include <random>
#include <set>
#include <iostream>

template < typename ps >
bool test_merge_sorted(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    using base_t = typename ps::base_type;
    // Allocate Memory
    base_t* test_out = reinterpret_cast<base_t*>(malloc((v2_data_count+v1_data_count)*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc((v2_data_count+v1_data_count)*sizeof(base_t)));
    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    
    memset(test_out, 0, (v2_data_count + v1_data_count)*sizeof(base_t));
    memset(ref_out, 0, (v2_data_count + v1_data_count)*sizeof(base_t));

    //Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());
    
    // Fill memory with sorted data
    std::set<base_t> set1, set2, set3;
    while(set1.size() < v1_data_count){
        base_t value = dist(rng);
        set1.insert(value);
        set3.insert(value);
    }

    while(set2.size() < v2_data_count){
        base_t value = dist(rng);
        set2.insert(value);
        set3.insert(value);
    }

    std::copy(set1.begin(), set1.end(), v1);
    std::copy(set2.begin(), set2.end(), v2);
    std::copy(set3.begin(), set3.end(), ref_out);

    typename tuddbs::merge_sorted<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    while(((state.p_Data1Ptr - v1 + ps::vector_element_count()) < v1_data_count) && ((state.p_Data2Ptr - v2 + ps::vector_element_count()) < v2_data_count)){
        tuddbs::merge_sorted<ps>{}(state);
        
        const size_t temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        const size_t temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;

    tuddbs::merge_sorted<ps>::flush(state);
    
    // Compare with reference result
    bool allOk = true;
    for(int i = 0; i < v2_data_count + v1_data_count; i++){
        allOk &= (ref_out[i] == test_out[i]);
        if(!allOk){
            break;
        }
    }

    free(v1);
    free(v2);
    free(test_out);
    free(ref_out);

    return allOk;
}

int main(){
    const int v1_count = 100;
    const int v2_count = 1000;
    bool allOk;
    std::cout << "Tesing merge_sorted" << std::endl;
    // INTEL - AVX512
    {
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX512 with uint64_t: ";
        allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;

    }{
        using ps = typename tsl::simd<int32_t, tsl::avx512>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX512 with int32_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;

    }{
        using ps = typename tsl::simd<uint16_t, tsl::avx512>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX512 with uint16_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<int8_t, tsl::avx512>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX512 with int8_t: ";
        std::cout.flush();
        bool allOk = test_merge_sorted<ps>(batch_size, 20, 15);
        allOk &= test_merge_sorted<ps>(batch_size, 15, 20);
        allOk &= test_merge_sorted<ps>(batch_size, 20, 20);
        std::cout << allOk << std::endl;
    }
    // INTEL AVX2
    {
        using ps = typename tsl::simd<uint64_t, tsl::avx2>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX2 with uint64_t: ";
        allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;

    }{
        using ps = typename tsl::simd<int32_t, tsl::avx2>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX2 with int32_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<uint16_t, tsl::avx2>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX2 with uint16_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<int8_t, tsl::avx2>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-AVX2 with int8_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, 20, 15);
        allOk &= test_merge_sorted<ps>(batch_size, 15, 20);
        allOk &= test_merge_sorted<ps>(batch_size, 20, 20);
        std::cout << allOk << std::endl;
    }
    // INTEL SSE
    {
        using ps = typename tsl::simd<uint64_t, tsl::sse>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with uint64_t: ";
        allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<int32_t, tsl::sse>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with int32_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<uint16_t, tsl::sse>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with uint16_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<int8_t, tsl::sse>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with int8_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, 20, 15);
        allOk &= test_merge_sorted<ps>(batch_size, 15, 20);
        allOk &= test_merge_sorted<ps>(batch_size, 20, 20);
        std::cout << allOk << std::endl;
    }
    // INTEL SCALAR
    {
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with uint64_t: ";
        allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<int32_t, tsl::scalar>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with int32_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<uint16_t, tsl::scalar>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with uint16_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, v1_count, v2_count);
        allOk &= test_merge_sorted<ps>(batch_size, v2_count, v1_count);
        allOk &= test_merge_sorted<ps>(batch_size, v1_count, v1_count);
        std::cout << allOk << std::endl;
    }{
        using ps = typename tsl::simd<int8_t, tsl::scalar>;
        const size_t batch_size = ps::vector_element_count();

        std::cout << "\t-SSE with int8_t: ";
        bool allOk = test_merge_sorted<ps>(batch_size, 20, 15);
        allOk &= test_merge_sorted<ps>(batch_size, 15, 20);
        allOk &= test_merge_sorted<ps>(batch_size, 20, 20);
        std::cout << allOk << std::endl;
    }

    std::cout << "merge_sorted result: " << allOk << std::endl;

    return 0;
}