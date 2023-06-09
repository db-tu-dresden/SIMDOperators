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

            /// @brief The alignment to check for in bytes.
            size_t alignment;
            /// @brief The offset of the pointer to desired alignment in bytes.
            size_t offset;

        public:
            Alignment(const void *ptr, size_t alignment) : ptr(ptr) {
                this->alignment = alignment;
                this->offset = reinterpret_cast<size_t>(ptr) % alignment;
            }

            /**
             * @brief Returns true if the pointer is aligned to the desired alignment.
             * 
             * @return true 
             * @return false 
             */
            bool isAligned() const {
                return offset == 0;
            }

            /**
             * @brief Returns the offset of the pointer to the desired alignment in bytes.
             * 
             * @return size_t 
             */
            size_t getOffset() const {
                return offset;
            }

            /**
             * @brief Returns the offset of the pointer to the desired alignment in elements.
             * 
             * @return size_t 
             */
            size_t getElementOffset() const {
                return offset / ProcessingStyle::vector_size_B();
            }

            /**
             * @brief Returns the number of elements until the pointer is aligned to the desired alignment.
             * 
             * @return size_t 
             */
            size_t getElementsUntilAlignment() const {
                return ((alignment - offset) % alignment) / sizeof(typename ProcessingStyle::base_type);
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