#include "merge_sorted.hpp"
#include "merge_sorted_comp.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <set>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std;
using ps = tsl::simd<uint64_t, tsl::avx2>;
using base_t = typename ps::base_type;

void printVec(base_t* vec, const size_t length, char* name){
    std::cout << name <<": [";
    for(int i = 0; i < length; i++){
        std::cout << vec[i] << ", ";
    }
    std::cout << "]" << endl;
}

void generate_data(char* filename, int element_count){
    //Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());

    // Use unordered set for faster generation
    unordered_set<base_t> set;
    std::cout << "Generate " << filename << " in progress" << std::endl;
    while(set.size() < element_count){
        base_t value = dist(rng);
        set.insert(value);
    }
    std::cout << "Sorting Data" << std::endl;

    // Put into Vector and sort
    std::vector<base_t> vec(set.begin(), set.end());
    std::sort(vec.begin(), vec.end());

    std::cout << "Begin writing into file" << std::endl;
    ofstream file(filename);
    for(auto i = vec.begin(); i != vec.end(); i++){
        file << *i << endl;
    }
    file.close();
}

void benchmark_merge_sorted(const size_t batch_size, const int v1_data_count, const int v2_data_count){
    // Allocate Memory
    base_t* test_out = reinterpret_cast<base_t*>(malloc((v2_data_count+v1_data_count)*sizeof(base_t)));
    base_t* v1 = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2 = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));
    base_t* ref_out = reinterpret_cast<base_t*>(malloc((v2_data_count+v1_data_count)*sizeof(base_t)));
    base_t* test_out_no_simd = reinterpret_cast<base_t*>(malloc((v2_data_count+v1_data_count)*sizeof(base_t)));
    base_t* v1_no_simd = reinterpret_cast<base_t*>(malloc(v1_data_count*sizeof(base_t)));
    base_t* v2_no_simd = reinterpret_cast<base_t*>(malloc(v2_data_count*sizeof(base_t)));

    memset(test_out, 0, (v2_data_count + v1_data_count)*sizeof(base_t));
    memset(test_out_no_simd, 0, (v2_data_count + v1_data_count)*sizeof(base_t));
    memset(ref_out, 0, (v2_data_count + v1_data_count)*sizeof(base_t));

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
    std::copy(v1, v1 + v1_data_count, v1_no_simd);
    std::copy(v2, v2 + v2_data_count, v2_no_simd);

    // No SIMD Benchmark
    size_t temp1, temp2;
    std::cout << "Start Test no SIMD:" << endl;
    merge_sorted_no_simd<ps>::State state2 = {.result_ptr = test_out_no_simd, .p_Data1Ptr = v1_no_simd, .p_CountData1 = batch_size, .p_Data2Ptr = v2_no_simd, .p_CountData2 = batch_size};
    auto start = chrono::high_resolution_clock::now();
    while(state2.p_Data1Ptr - v1_no_simd < v1_data_count && state2.p_Data2Ptr - v2_no_simd < v2_data_count){
        merge_sorted_no_simd<ps>{}(state2);

        temp1 = v1_no_simd + v1_data_count - state2.p_Data1Ptr;
        temp2 = v2_no_simd + v2_data_count - state2.p_Data2Ptr;

        state2.p_CountData1 = std::min(batch_size, temp1);
        state2.p_CountData2 = std::min(batch_size, temp2);
    }
    state2.p_CountData1 = v1_no_simd + v1_data_count - state2.p_Data1Ptr;
    state2.p_CountData2 = v2_no_simd + v2_data_count - state2.p_Data2Ptr;
    merge_sorted_no_simd<ps>::flush(state2);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration without SIMD: " << duration.count() << " microseconds" << endl;
    
    bool allOk = true;
    for(int i = 0; i < v2_data_count + v1_data_count; i++){
        allOk &= (ref_out[i] == test_out_no_simd[i]);
        if(!allOk){
            std::cout << "No SIMD: smth went wrong at index: " << i << endl;
            printVec(&test_out_no_simd[i], batch_size, "TestData");
            printVec(&ref_out[i], batch_size, "RefData");
            break;
        }
    }

    // SIMD Benchmark
    tuddbs::merge_sorted<ps>::State state = {.result_ptr = test_out, .p_Data1Ptr = v1, .p_CountData1 = batch_size, .p_Data2Ptr = v2, .p_CountData2 = batch_size};
    std::cout << "\nStart Test SIMD:" << endl;
    start = chrono::high_resolution_clock::now();
    while(((state.p_Data1Ptr - v1 + ps::vector_element_count()) < v1_data_count) && ((state.p_Data2Ptr - v2 + ps::vector_element_count()) < v2_data_count)){
        tuddbs::merge_sorted<ps>{}(state);
        
        temp1 = v1 + v1_data_count - state.p_Data1Ptr;
        temp2 = v2 + v2_data_count - state.p_Data2Ptr;

        state.p_CountData1 = std::min(batch_size, temp1);
        state.p_CountData2 = std::min(batch_size, temp2);
    }
    state.p_CountData1 = v1 + v1_data_count - state.p_Data1Ptr;
    state.p_CountData2 = v2 + v2_data_count - state.p_Data2Ptr;
    auto s1 = chrono::high_resolution_clock::now();
    tuddbs::merge_sorted<ps>::flush(state);
    end = chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(end - s1);
    auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(s1 - start);
    std::cout << "Duration with SIMD:\tBatch Loop: " << d2.count() << "μs\tFlush: " << d1.count() << "μs\tTotal: " << duration.count() << "μs" <<std::endl; 

    // Compare with reference result
    allOk = true;
    for(int i = 0; i < v2_data_count + v1_data_count; i++){
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
    std::free(v1_no_simd);
    std::free(v2_no_simd);
    std::free(test_out_no_simd);
    std::free(ref_out);
}

int main(){
    const size_t batch_size = 4 * ps::vector_element_count();
    
    const int v1_count = 5 * 1024 * 1024 * 1024 / 8;
    //const int v2_count = 3 * 1024 * 1024 * 1024 / 8;

    generate_data("v1_data", v1_count);
    generate_data("v2_data", v1_count);
    for(int i = 0; i < 10; i++){
        benchmark_merge_sorted(batch_size, v1_count, v1_count);
        std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
    }

    return 0;
}