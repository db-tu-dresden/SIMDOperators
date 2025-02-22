#include <immintrin.h>
#include <iostream>
#include <cstdint>
#include <iomanip>
#include <bitset>

size_t intersect_sorted(uint64_t* p_OutPtr, uint64_t* p_Data1Ptr,
                        uint64_t* p_Data2Ptr,
                        const size_t l_elements, const size_t r_elements) {
    const uint64_t* const endInPosL = p_Data1Ptr + l_elements;
    const uint64_t* const endInPosR = p_Data2Ptr + r_elements;
    const uint64_t* resStart = p_OutPtr;

    __m256i data1Vector;
    __m256i data2Vector;
    int mask = 0;
    int mask_greater_than = 0;
    data1Vector = _mm256_loadu_si256(
        (__m256i*)p_Data1Ptr);  // Load the first 4 values of the left column
    data2Vector = _mm256_loadu_si256(
        (__m256i*)p_Data2Ptr);  // Load the first 4 values of the right column
    int full_hit = _mm256_movemask_pd(
        (__m256d)(_mm256_cmpeq_epi64(data1Vector, data1Vector)));

    while (p_Data1Ptr < endInPosL && p_Data2Ptr <= endInPosR - 4) {
        mask = _mm256_movemask_pd(
            (__m256d)(_mm256_cmpeq_epi64(data2Vector, data1Vector)));

        mask_greater_than = _mm256_movemask_pd(
            (__m256d)(_mm256_cmpgt_epi64(data1Vector, data2Vector)));

        if (mask != 0) {
            std::cout << "Adding " << *p_Data1Ptr << std::endl;
            *p_OutPtr = *p_Data1Ptr; 
            p_OutPtr++;
        }
        if ((mask_greater_than) == 0) {
            std::cout << "Advancing LHS by 1" << std::endl;
            p_Data1Ptr++;
            data1Vector = _mm256_set1_epi64x(*p_Data1Ptr);

        } else {
            if ((mask_greater_than) == full_hit) {
                std::cout << "All greater, advancing LHS by 4" << std::endl;
                p_Data2Ptr += 4;
                data2Vector = _mm256_loadu_si256((__m256i*)p_Data2Ptr);

            } else {
                std::cout << "Advancing LHS by 1" << std::endl;
                p_Data1Ptr++;
                data1Vector = _mm256_set1_epi64x(*p_Data1Ptr);
                const size_t adv = __builtin_popcount(mask_greater_than);
                std::cout << "Advancing RHS by " << adv << std::endl;
                p_Data2Ptr += adv;
                data2Vector = _mm256_loadu_si256((__m256i*)p_Data2Ptr);
            }
        }
        std::cout << "Next iteration LHS condition: " << std::boolalpha << (p_Data1Ptr < endInPosL) << " Elements left: " << endInPosL - p_Data1Ptr << std::endl;
        std::cout << "Next iteration RHS condition: " << std::boolalpha << (p_Data2Ptr < (endInPosR - 4)) << " Elements left: " << endInPosR - p_Data2Ptr << std::endl;
    }
    std::cout << "Scalar remainder" << std::endl;
    while ( p_Data1Ptr < endInPosL && p_Data2Ptr < endInPosR ) {
        if (*p_Data1Ptr == *p_Data2Ptr ) {
            std::cout << "Advancing both" << std::endl;
            *p_OutPtr++ = *p_Data1Ptr++;
            p_Data2Ptr++;
        } else if (*p_Data1Ptr < *p_Data2Ptr) {
            std::cout << "Advancing left" << std::endl;
            ++p_Data1Ptr;
        } else {
            std::cout << "Advancing right" << std::endl;
            ++p_Data2Ptr;
        }
    }
    return p_OutPtr - resStart;
}

int main() {
    uint64_t a[] = {1,2,3,4,5,6,7,8,9};
    uint64_t b[] = {2,4,5,6,8,9};
    uint64_t* res = static_cast<uint64_t*>(malloc(10 * sizeof(uint64_t)));

    const size_t res_elements = intersect_sorted( res, a, b, 9, 6);
    std::cout << "Results: " << res_elements << std::endl;
    for ( size_t i = 0; i < res_elements; ++i ) {
        std::cout << "res[" << i << "]: " << res[i] << std::endl;
    }

    free(res);
}