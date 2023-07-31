#include "../src/SIMDOperators/operators/intersect_sorted.hpp"
#include "intersect_sorted_comp.hpp"
#include "help_functions.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <chrono>

using namespace std;
using ps = tsl::simd<uint64_t, tsl::avx2>;

template <typename ps>
void benchmark_no_simd(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    using base_t = typename ps::base_type;

    int result_count;
    (v2_data_count < v1_data_count)? result_count = v2_data_count : result_count = v1_data_count;

    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    base_t* test_out = reinterpret_cast<base_t*>(malloc(result_count*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc(result_count*sizeof(base_t)));
    
    memset(test_out, 0, result_count*sizeof(base_t));
    memset(ref_out, 0, result_count*sizeof(base_t));

    // Fill memory with dataset for benchmark
    std::set<base_t> set1, set2;
    std::ifstream file1("v1_data");
    base_t read_value;
    while(file1 >> read_value){
        set1.insert(read_value);
    }
    file1.close();

    std::ifstream file2("v2_data");
    while(file2 >> read_value){
        set2.insert(read_value);
    }
    file2.close();
    std::vector<base_t> intersect;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(intersect));
    
    std::copy(set1.begin(), set1.end(), v1);
    std::copy(set2.begin(), set2.end(), v2);
    std::copy(intersect.begin(), intersect.end(), ref_out);

    //Begin Benchmark
    std::cout << "Start Test no SIMD:" << endl;
    typename intersect_sorted_no_simd<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    auto start = chrono::high_resolution_clock::now();
    while(state.p_Data1Ptr - v1 < v1_data_count && state.p_Data2Ptr - v2 < v2_data_count){
        intersect_sorted_no_simd<ps>{}(state);

        size_t temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        size_t temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;
    intersect_sorted_no_simd<ps>::flush(state);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration without SIMD: " << duration.count() << "μs" << endl;
    
    bool allOk = true;
    for(int i = 0; i < result_count; i++){
        allOk &= (ref_out[i] == test_out[i]);
        if(!allOk){
            std::cout << "No SIMD: smth went wrong at index: " << i << endl;
            printVec<base_t>(&test_out[i], batch_size, "TestData");
            printVec<base_t>(&ref_out[i], batch_size, "RefData");
            break;
        }
    }

    std::free(v1);
    std::free(v2);
    std::free(test_out);
    std::free(ref_out);
}

template <typename ps>
void benchmark_intersect_sorted(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    using base_t = typename ps::base_type;
    int result_count;
    (v2_data_count < v1_data_count)? result_count = v2_data_count : result_count = v1_data_count;

    // Allocate Memory
    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    base_t* test_out = reinterpret_cast<base_t*>(malloc(result_count*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc(result_count*sizeof(base_t)));

    memset(test_out, 0, result_count*sizeof(base_t));
    memset(ref_out, 0, result_count*sizeof(base_t));

    // Fill memory with dataset for benchmark
    std::set<base_t> set1, set2;
    ifstream file1("v1_data");
    base_t read_value;
    while(file1 >> read_value){
        set1.insert(read_value);
    }
    file1.close();

    ifstream file2("v2_data");
    while(file2 >> read_value){
        set2.insert(read_value);
    }
    file2.close();
    std::vector<base_t> intersect;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(intersect));
    
    std::copy(set1.begin(), set1.end(), v1);
    std::copy(set2.begin(), set2.end(), v2);
    std::copy(intersect.begin(), intersect.end(), ref_out);

    // SIMD Benchmark
    typename tuddbs::intersect_sorted<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    std::cout << "\nStart Test SIMD:" << endl;
    auto start = chrono::high_resolution_clock::now();
    while(((state.p_Data1Ptr - v1 + ps::vector_element_count()) < v1_data_count) && ((state.p_Data2Ptr - v2 + ps::vector_element_count()) < v2_data_count)){
        tuddbs::intersect_sorted<ps>{}(state);
        
        size_t temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        size_t temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;
    auto s1 = chrono::high_resolution_clock::now();
    tuddbs::intersect_sorted<ps>::flush(state);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(end - s1);
    auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(s1 - start);
    std::cout << "Duration with SIMD:\tBatch Loop: " << d2.count() << "μs\tFlush: " << d1.count() << "μs\tTotal: " << duration.count() << "μs" <<std::endl; 

    // Compare with reference result
    bool allOk = true;
    for(int i = 0; i < result_count; i++){
        allOk &= (ref_out[i] == test_out[i]);
        if(!allOk){
            std::cout << "SIMD: smth went wrong" << endl;
            printVec(&test_out[i], batch_size, "TestData");
            printVec(&ref_out[i], batch_size, "RefData");
            break;
        }
    }

    std::free(v1);
    std::free(v2);
    std::free(test_out);
    std::free(ref_out);
}

template <typename ps>
void benchmark_wrapper(){
    const size_t batch_size = 4 * ps::vector_element_count();
    const int count = 2 * 1024 * 1024 * 1024 / 8;

    std::cout << "Batchsize: " << batch_size << std::endl;
    benchmark_merge_sorted<ps>(batch_size, count, count);
    std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
}

int main(){
    using ps = tsl::simd<uint64_t, tsl::avx2>;
    std::cout << "Starting merge_sorted Benchmark..." << std::endl;
    const size_t batch_size = 4 * ps::vector_element_count();
    const int count = 2 * 1024 * 1024 * 1024 / 8;
    
    benchmark_no_simd<ps>(batch_size, count, count);
    benchmark_wrapper<ps>();

    return 0;
}