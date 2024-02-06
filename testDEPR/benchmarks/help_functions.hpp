#include <iostream>
#include <fstream>
#include <random>
#include <unordered_set>
#include <algorithm>

template<typename base_t>
void printVec(base_t* vec, const size_t length, char* name){
    std::cout << name <<": [";
    for(int i = 0; i < length; i++){
        std::cout << vec[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

template <typename base_t>
void generate_data(char* filename, int element_count){
    //Random number generator
    auto seed = std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<base_t> dist(1, std::numeric_limits<base_t>::max());

    // Use unordered set for faster generation
    std::unordered_set<base_t> set;
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
    std::ofstream file(filename);
    for(auto i = vec.begin(); i != vec.end(); i++){
        file << *i << std::endl;
    }
    file.close();
}