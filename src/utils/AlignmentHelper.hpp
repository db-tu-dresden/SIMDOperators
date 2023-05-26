#ifndef SRC_UTILS_ALIGNMENTHELPER_HPP
#define SRC_UTILS_ALIGNMENTHELPER_HPP


#include <cstddef>
#include <tslintrin.hpp>

namespace tuddbs{


    template<typename ProcessingStyle = tsl::simd<uint8_t, tsl::scalar>>
    struct AlignmentHelper {

        class Alignment {
        private:
            const void *ptr;

            size_t alignment;
            size_t offset;

        public:
            Alignment(const void *ptr, size_t alignment) : ptr(ptr) {
                this->alignment = alignment;
                this->offset = reinterpret_cast<size_t>(ptr) % alignment;
            }

            bool isAligned() const {
                return offset == 0;
            }

            size_t getOffset() const {
                return offset;
            }

            size_t getElementOffset() const {
                return offset / ProcessingStyle::vector_size_B();
            }

            size_t getAlignment() const {
                return alignment;
            }

            const void *getPtr() const {
                return ptr;
            }

            const void *getFirstAlignedPtrWithin() const {
                size_t ptrValue = reinterpret_cast<size_t>(ptr);
                return reinterpret_cast<const void *>(ptrValue + (ptrValue % alignment));
            }

            bool operator==(const Alignment &other) const {
                return offset == other.offset;
            }

        };


        static const Alignment getAlignment(const void *ptr, size_t alignment) {
            return Alignment(ptr, alignment);
        }

        static const Alignment getAlignment(const void *ptr) {
            return Alignment(ptr, ProcessingStyle::vector_size_B());
        }

    };
}; //namespace tuddbs
#endif //SRC_UTILS_ALIGNMENTHELPER_HPP