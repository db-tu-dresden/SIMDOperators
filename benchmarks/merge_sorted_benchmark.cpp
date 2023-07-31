#include "../src/SIMDOperators/operators/merge_sorted.hpp"
#include "merge_sorted_comp.hpp"
#include "help_functions.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <chrono>

using namespace std;
using ps = tsl::simd<uint64_t, tsl::avx2>;
using base_t = typename ps::base_type;

void benchmark_merge_sorted_no_simd(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    base_t* test_out = reinterpret_cast<base_t*>(malloc((v1_data_count + v2_data_count)*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc((v1_data_count + v2_data_count)*sizeof(base_t)));
    
    memset(test_out, 0, (v1_data_count + v2_data_count)*sizeof(base_t));
    memset(ref_out, 0, (v1_data_count + v2_data_count)*sizeof(base_t));

    // Fill memory with dataset for benchmark
    std::set<base_t> set1;
    ifstream file1("v1_data");
    base_t read_value;
    base_t* vec_beginning = v1;
    while(file1 >> read_value){
        *v1 = read_value;
        v1++;
        set1.insert(read_value);
    }
    file1.close();
    v1 = vec_beginning;

    vec_beginning = v2;
    ifstream file2("v2_data");
    while(file2 >> read_value){
        *v2 = read_value;
        v2++;
        set1.insert(read_value);
    }
    file2.close();
    v2 = vec_beginning;
    std::copy(set1.begin(), set1.end(), ref_out);

    //Begin Benchmark
    std::cout << "Start Test no SIMD:" << endl;
    merge_sorted_no_simd<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    auto start = chrono::high_resolution_clock::now();
    while(state.p_Data1Ptr - v1 < v1_data_count && state.p_Data2Ptr - v2 < v2_data_count){
        merge_sorted_no_simd<ps>{}(state);

        size_t temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        size_t temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;
    merge_sorted_no_simd<ps>::flush(state);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration without SIMD: " << duration.count() << "μs" << endl;
    
    bool allOk = true;
    for(int i = 0; i < (v1_data_count + v2_data_count); i++){
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

void benchmark_merge_sorted(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    // Allocate Memory
    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    base_t* test_out = reinterpret_cast<base_t*>(malloc((v1_data_count + v2_data_count)*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc((v1_data_count + v2_data_count)*sizeof(base_t)));

    memset(test_out, 0, (v1_data_count + v2_data_count)*sizeof(base_t));
    memset(ref_out, 0, (v1_data_count + v2_data_count)*sizeof(base_t));

    // Fill memory with dataset for benchmark
    std::set<base_t> set1;
    ifstream file1("v1_data");
    base_t read_value;
    base_t* vec_beginning = v1;
    while(file1 >> read_value){
        *v1 = read_value;
        v1++;
        set1.insert(read_value);
    }
    file1.close();
    v1 = vec_beginning;

    vec_beginning = v2;
    ifstream file2("v2_data");
    while(file2 >> read_value){
        *v2 = read_value;
        v2++;
        set1.insert(read_value);
    }
    file2.close();
    v2 = vec_beginning;
    std::copy(set1.begin(), set1.end(), ref_out);

    // SIMD Benchmark
    tuddbs::merge_sorted<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    std::cout << "\nStart Test SIMD:" << endl;
    auto start = chrono::high_resolution_clock::now();
    while(((state.p_Data1Ptr - v1 + ps::vector_element_count()) < v1_data_count) && ((state.p_Data2Ptr - v2 + ps::vector_element_count()) < v2_data_count)){
        tuddbs::merge_sorted<ps>{}(state);
        
        size_t temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        size_t temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;
    auto s1 = chrono::high_resolution_clock::now();
    tuddbs::merge_sorted<ps>::flush(state);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(end - s1);
    auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(s1 - start);
    std::cout << "Duration with SIMD:\tBatch Loop: " << d2.count() << "μs\tFlush: " << d1.count() << "μs\tTotal: " << duration.count() << "μs" <<std::endl; 

    // Compare with reference result
    bool allOk = true;
    for(int i = 0; i < (v1_data_count + v2_data_count); i++){
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

int main(){
    std::cout << "Starting merge_sorted Benchmark..." << std::endl;
    size_t batch_size = ps::vector_element_count();
    const int count = 5 * 1024 * 1024 * 1024 / 8;
    
    for(int i = 1; i < 8; i++){
        batch_size = batch_size * 10;
        std::cout << "Batchsize: " << batch_size << std::endl;
        
        benchmark_merge_sorted_no_simd(batch_size, count, count);
        benchmark_merge_sorted(batch_size, count, count);
        std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
    }

    return 0;
}