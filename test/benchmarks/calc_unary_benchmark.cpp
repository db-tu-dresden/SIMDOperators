#include "../../src/SIMDOperators/operators/calc_unary.hpp"
#include "help_functions.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <chrono>

using namespace std;

template< typename ps , template <typename ...> typename Operator>
void benchmark_merge_sorted(const size_t batch_size, const int data_count){
    using base_t = typename ps::base_type;

    // Allocate Memory
    base_t* vec = reinterpret_cast<base_t*>(malloc(data_count*sizeof(base_t)));
    base_t* result = reinterpret_cast<base_t*>(malloc(data_count*sizeof(base_t)));

    memset(result, 0, (data_count)*sizeof(base_t));

    // Fill memory with dataset for benchmark
    std::set<base_t> set1;
    ifstream file1("v1_data");
    base_t read_value;
    base_t* vec_beginning = vec;
    while(file1 >> read_value){
        *vec++ = read_value;
    }
    file1.close();
    vec = vec_beginning;

    // SIMD Benchmark
    typename tuddbs::calc_unary<ps, Operator>::State state = {.result_ptr = result, .p_Data1Ptr = vec, .p_CountData1 = batch_size};
    std::cout << "\nStart Test SIMD:" << endl;
    auto start = chrono::high_resolution_clock::now();
    while((state.p_Data1Ptr - vec + ps::vector_element_count()) < data_count){
        tuddbs::calc_unary<ps, Operator>{}(state);

        const size_t t = vec + data_count - state.p_Data1Ptr;

        state.p_CountData1 = std::min(batch_size, t);
    }
    state.p_CountData1 = vec + data_count - state.p_Data1Ptr;
    tuddbs::calc_unary<ps, Operator>::flush(state);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration with SIMD: " << duration.count() << "μs" << endl;

    std::free(vec);
    std::free(result);
}

template< typename ps , template <typename ...> typename Operator>
void benchmark_wrapper(){
    const size_t batch_size = 4 * ps::vector_element_count();
    const int count = 5 * 1024 * 1024 * 1024 / 8;

    std::cout <<  tsl::type_name<ps>() << "\nBatchsize: " << batch_size << std::endl;
    benchmark_merge_sorted<ps, Operator>(batch_size, count);
    std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
}

int main(){
    std::cout << "Starting merge_sorted Benchmark..." << std::endl;
    std::cout << "Operation: Invert" << std::endl;
    benchmark_wrapper<tsl::simd<uint64_t, tsl::avx512>, tsl::functors::inv>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::avx2>, tsl::functors::inv>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::sse>, tsl::functors::inv>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::scalar>, tsl::functors::inv>();
    std::cout << "Operation: hadd" << std::endl;
    benchmark_wrapper<tsl::simd<uint64_t, tsl::avx512>, tsl::functors::hadd>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::avx2>, tsl::functors::hadd>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::sse>, tsl::functors::hadd>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::scalar>, tsl::functors::hadd>();
    std::cout << "Operation: unequal_zero" << std::endl;
    benchmark_wrapper<tsl::simd<uint64_t, tsl::avx512>, tsl::functors::unequal_zero>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::avx2>, tsl::functors::unequal_zero>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::sse>, tsl::functors::unequal_zero>();
    benchmark_wrapper<tsl::simd<uint64_t, tsl::scalar>, tsl::functors::unequal_zero>();

    return 0;
}