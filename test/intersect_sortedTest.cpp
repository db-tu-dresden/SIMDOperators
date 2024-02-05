#include "../src/SIMDOperators/operators/intersect_sorted.hpp"
#include <algorithm>
#include <random>
#include <set>
#include <vector>
#include <iostream>

template < typename ps >
bool test_intersect_sorted(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    using base_t = typename ps::base_type;
    
    int result_count;
    (v2_data_count < v1_data_count)? result_count = v2_data_count : result_count = v1_data_count;
    // Allocate Memory
    base_t* test_out = reinterpret_cast<base_t*>(malloc((result_count)*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc((result_count)*sizeof(base_t)));
    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    
    memset(test_out, 0, (result_count)*sizeof(base_t));
    memset(ref_out, 0, (result_count)*sizeof(base_t));

    //Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    
    // Fill memory with sorted data
    std::set<base_t> set1, set2;
    if constexpr(std::is_integral_v<base_t>){
        std::uniform_int_distribution<base_t> dist(std::numeric_limits<base_t>::min(), std::numeric_limits<base_t>::max());
        while(set1.size() < v1_data_count){
            base_t value = dist(rng);
            set1.insert(value);
        }
        while(set2.size() < v2_data_count){
            base_t value = dist(rng);
            set2.insert(value);
        }
    }else{
        std::uniform_real_distribution<base_t> dist(std::numeric_limits<base_t>::min(), std::numeric_limits<base_t>::max());
        while(set1.size() < v1_data_count){
            base_t value = dist(rng);
            set1.insert(value);
        }
        while(set2.size() < v2_data_count){
            base_t value = dist(rng);
            set2.insert(value);
        }
    }
    std::vector<base_t> intersect;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(intersect));

    std::copy(set1.begin(), set1.end(), v1);
    std::copy(set2.begin(), set2.end(), v2);
    std::copy(intersect.begin(), intersect.end(), ref_out);

    typename tuddbs::intersect_sorted<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    while(((state.p_Data1Ptr - v1 + ps::vector_element_count()) < v1_data_count) && ((state.p_Data2Ptr - v2 + ps::vector_element_count()) < v2_data_count)){
        tuddbs::intersect_sorted<ps>{}(state);
        
        const size_t temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        const size_t temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;

    tuddbs::intersect_sorted<ps>::flush(state);
    
    // Compare with reference result
    bool allOk = true;
    for(int i = 0; i < result_count; i++){
        allOk &= (ref_out[i] == test_out[i]);
    }

    free(v1);
    free(v2);
    free(test_out);
    free(ref_out);

    return allOk;
}

template<typename ps>
bool do_test(){
    const int number_count = std::pow(2, sizeof(typename ps::base_type)*8);
    const int element_count = ps::vector_element_count();
    int v1_count = 100000;
    int v2_count = 1000000;
    if(v1_count > number_count){
        v1_count = number_count;
    }if(v2_count > number_count){
        v2_count = number_count;
    }

    bool allOk = true;
    allOk &= test_intersect_sorted<ps>(element_count, v1_count, v2_count);
    allOk &= test_intersect_sorted<ps>(element_count, v2_count, v1_count);
    allOk &= test_intersect_sorted<ps>(element_count, v2_count, v2_count);
    std::cout << tsl::type_name<typename ps::base_type>() << " : " << allOk << std::endl;
    return allOk;
}

template<typename tsl_simd>
bool test_intersect_sorted_wrapper(){
    bool allOk = true;
    std::cout << "Using: " << tsl::type_name<tsl_simd>() <<std::endl;
    allOk &= do_test<tsl::simd<uint8_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<int8_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<uint16_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<int16_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<uint32_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<int32_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<uint64_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<int64_t, tsl_simd>>();
    allOk &= do_test<tsl::simd<float, tsl_simd>>();
    allOk &= do_test<tsl::simd<double, tsl_simd>>();
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    return allOk;
}

int main()
{
    bool allOk = true;
    std::cout << "Testing intersect_sorted...\n";
    std::cout.flush();
    allOk &= test_intersect_sorted_wrapper<tsl::scalar>();
    allOk &= test_intersect_sorted_wrapper<tsl::sse>();
    allOk &= test_intersect_sorted_wrapper<tsl::avx2>();
    allOk &= test_intersect_sorted_wrapper<tsl::avx512>();
    std::cout << "Complete Result: " << allOk << std::endl;
    return 0;
}