
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <random>
#include <SIMDOperators/SIMDOperators.h>
#include <SIMDOperators/datastructure/column.hpp>



using namespace std;
using namespace tuddbs;
using namespace std::chrono;

struct state_t {
    uint64_t dataSize;
    uint64_t predicate;
    uint64_t offset;
    uint64_t element_count;
    bool aligned;
    string processing_style;
    nanoseconds duration;
    string operatorString;

    /// to stringstream
    friend ostream& operator<<(ostream& os, const state_t& state){
        os << state.operatorString << "," << state.processing_style << "," << state.dataSize << "," << state.offset << "," 
        << state.aligned;
        // os << state.dataSize << "," << state.predicate << "," << state.offset << "," << state.element_count << "," << state.aligned << "," << state.processing_style << "," << state.duration.count();
        return os;
    }
};

struct Config {
    static state_t state;
    static ofstream outStream;
    static vector<uint64_t> dataSizes;
    static size_t runs;
};

template<typename PS, typename alignedOp, typename unalignedOp>
class Benchmark {
    using self = Benchmark<PS, alignedOp, unalignedOp>;
    using base_type = typename PS::base_type;
    constexpr static uint64_t vector_size = PS::vector_size_B();
    constexpr static uint64_t vector_element_count = PS::vector_element_count();
    using col_type = Column<base_type>;
    using col_ptr = std::shared_ptr<col_type>;
    using const_col_ptr = std::shared_ptr<const col_type>;
    

    template<typename ... Args>
    static void _fire_aligned(const const_col_ptr& column, bool aligned, Args...args){
        Config::state.aligned = aligned;
        cout << "processing " << Config::state.processing_style << " with " << Config::state.dataSize << " bytes, " 
        << (aligned ? "aligned" : "unaligned") << ", offset " << Config::state.offset << endl;
        vector<nanoseconds> durations(Config::runs);
        Config::outStream << Config::state;
        for(size_t i = 0; i < Config::runs; ++i){
            nanoseconds duration;
            col_ptr res;
            if(aligned){
                auto begin = high_resolution_clock::now();
                res = alignedOp::apply(column, args...);
                auto end = high_resolution_clock::now();
                duration = duration_cast<nanoseconds>(end - begin);
            } else {
                auto begin = high_resolution_clock::now();
                res = unalignedOp::apply(column, args...);
                auto end = high_resolution_clock::now();
                duration = duration_cast<nanoseconds>(end - begin);
            }
            Config::outStream << "," << duration.count();
            durations.push_back(duration);
            cout << "Run " << i+1 << "/" << Config::runs << ": duration " << duration.count() << " ns";
            cout << " (result: " << res.get()->getPopulationCount() << ")" << endl;
        }
        Config::outStream << endl;

        sort(durations.begin(), durations.end());
        /// delete first and last element
        durations.erase(durations.begin());
        durations.pop_back();
        nanoseconds sum = accumulate(durations.begin(), durations.end(), nanoseconds(0));
        nanoseconds avg = sum / durations.size();
        cout << "avg: " << avg.count() << " ns" << endl;

    }

    template<typename ... Args>
    static void _fire_offset(const const_col_ptr& column, uint64_t element_count, uint64_t offset, Args...args){
        Config::state.offset = offset;
        const_col_ptr off_column = column.get()->chunk(offset, element_count);
        _fire_aligned(off_column, false, args...);
        _fire_aligned(off_column, true, args...);
    }

    template<typename ... Args>
    static void _fire_dataSize(const const_col_ptr& column, uint64_t dataSize, Args...args) {
        Config::state.dataSize = dataSize;
        uint64_t element_count = dataSize / sizeof(base_type);

        auto col = column->chunk(0, element_count + vector_element_count);
        
        for(uint64_t i = 0; i < vector_element_count; ++i){
            _fire_offset(col, element_count, i, args...);
        }
    }

public:
    template<typename ... Args>
    static void fire(Args...args){

        uint64_t element_count = *std::max_element(Config::dataSizes.begin(), Config::dataSizes.end()) / sizeof(base_type) + vector_element_count * 2;
        
        /// create column
        col_ptr col = col_type::create(element_count, vector_size);
        col->setPopulationCount(element_count);
        std::mt19937 engine(0xFF00FF00FF00);
        base_type upper_bound = (base_type) 0xFFFFFFFF;
        std::uniform_int_distribution< base_type > dist( 0, upper_bound );

        /// fill column
        for (uint64_t i = 0; i < element_count + vector_element_count; ++i) {
            col.get()->getRawDataPtr()[i] = dist(engine);
        }

        for(auto dataSize : Config::dataSizes){
            _fire_dataSize(col, dataSize + sizeof(base_type), args...);
        }
        cout << "Fire finished" << endl;
    }
};



ofstream Config::outStream;
vector<uint64_t> Config::dataSizes;
state_t Config::state;
size_t Config::runs = 10;

int main(){
    #ifdef DEBUG
    cout << "DEBUG MODE" << endl;
    #endif

    // using ps = typename tsl::simd<uint64_t, tsl::avx512>;

    // auto col = Column<uint64_t>::create(100, ps::vector_size_B());
    // col->setPopulationCount(100);
    // // fill column
    // {
    //     auto data = col.get()->getData();
    //     for (int i = 0; i < col.get()->getLength(); ++i) {
    //         data[i] = i;
    //     }
    // }


    // // print column
    // {
    //     auto data = col.get()->getData();
    //     for (int i = 0; i < col.get()->getLength(); ++i) {
    //         cout << data[i] << " ";
    //     }
    //     cout << endl;
    // }

    uint64_t kb = 1024;
    uint64_t mb = 1024 * kb;

    /// Timestamp [dd.mm.yyyy hh:mm:ss] using chrono timepoint

    auto now = chrono::system_clock::now();
    auto now_c = chrono::system_clock::to_time_t(now);
    auto timestamp = put_time(localtime(&now_c), "%d_%m_%Y__%H_%M_%S");
    stringstream ss;
    ss << timestamp;
    string st(ss.str());
    // cout << "Timestamp: " << st << endl;
    // cout << "Timestamp: " << put_time(localtime(&now_c), "%d_%m_%Y__%H_%M_%S") << endl;

    // return 0;



    /// Start benchmark
    Config::outStream.open("benchmark_" + st + ".csv");
    // Config::dataSizes = vector{128 * kb, 256 * kb, 512 * kb, 1 * mb, 2 * mb, 4 * mb, 8 * mb, 16 * mb, 32 * mb, 64 * mb};
    Config::dataSizes = vector{128 * kb, 256 * kb, 512 * kb, 1 * mb, 2 * mb, 4 * mb, 8 * mb};
    // Config::dataSizes = vector{4 * mb, 8 * mb, 16 * mb};
    Config::state = state_t();
    Config::state.operatorString = "Select<greater_than>";

    // for(uint64_t i = 0; i < 100; i++)
    {
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
        Config::state.processing_style = "scalar";
        Benchmark<ps, tuddbs::select<ps, tsl::functors::greater_than>, tuddbs::select_unaligned<ps, tsl::functors::greater_than>>::fire(((uint64_t) 0xFFFFFFFF) / 2);
    }
    {
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;
        Config::state.processing_style = "avx512";
        Benchmark<ps, tuddbs::select<ps, tsl::functors::greater_than>, tuddbs::select_unaligned<ps, tsl::functors::greater_than>>::fire(((uint64_t) 0xFFFFFFFF) / 2);
        cout << "Benchmark finished" << endl;
    }
    {
        using ps = typename tsl::simd<uint64_t, tsl::avx2>;
        Config::state.processing_style = "avx2";
        Benchmark<ps, tuddbs::select<ps, tsl::functors::greater_than>, tuddbs::select_unaligned<ps, tsl::functors::greater_than>>::fire(((uint64_t) 0xFFFFFFFF) / 2);
        cout << "Benchmark finished" << endl;
    }


    Config::outStream.close();
}