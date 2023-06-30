#ifndef SRC_OPERATORS_PROJECT_HPP
#define SRC_OPERATORS_PROJECT_HPP

#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>

namespace tuddbs{
    template<typename ProcessingStyle>
    class project {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        template<typename batchps>
        class batch {
            // to make it accessable by project class
            friend class project<ProcessingStyle>;
            
            using reg_t = typename batchps::register_type;
            using mask_t = typename batchps::mask_type;
            using imask_t = typename batchps::imask_type;
            static const std::size_t VectorSizeInBits = batchps::target_extension::template types<base_type>::default_size_in_bits::value;
            using offset_register_type = std::array<typename batchps::offset_base_register_type, sizeof(offset_t)/sizeof(base_type)>;

            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            static size_t apply(base_type * result, const base_type * column, const base_type * positions, const size_t& vector_count){
                /// output data pointer
                // base_type * output_data = result.get()->getRawDataPtr();
                base_type * output_data = result;
                /// input data pointer
                // base_type const * input_data = column.get()->getRawDataPtr();
                base_type const * input_data = column;

                /// input positions pointer
                // base_type const * position_data = positions.get()->getRawDataPtr();
                base_type const * position_data = positions;

                size_t overall_count = 0;

                for (size_t i = 0; i < vector_count; ++i) {
                    /// load data into vector register
                    reg_t data_vector = tsl::load<batchps>(position_data);
                    /// gather the corresponding data
                    reg_t result_vector = tsl::gather<batchps, batchps>(input_data, data_vector);
                    /// store the resulting data
                    tsl::storeu<batchps>(output_data, result_vector);
                    /// increment the position vector
                    position_data += batchps::vector_element_count();
                    /// increment the output data pointer
                    output_data += batchps::vector_element_count();
                    /// increment the overall count
                    overall_count += batchps::vector_element_count();
                }
                return overall_count;

            }

            // MSV_CXX_ATTRIBUTE_FORCE_INLINE
            // static 
        };

    public:

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr apply(const_col_ptr column, const_col_ptr positions){
            /// Get the alignment of the positions column
            typename AlignmentHelper<ps>::Alignment alignment = AlignmentHelper<ps>::getAlignment(positions->getRawDataPtr());
            size_t alignment_elements;
            if (positions->getPopulationCount() < alignment.getElementsUntilAlignment()) {
                alignment_elements = positions->getPopulationCount();
            } else {
                alignment_elements = alignment.getElementsUntilAlignment();
            }

            Column<base_type> * result = new Column<base_type>(positions->getPopulationCount(), ps::vector_size_B());

            auto result_ptr = result->getRawDataPtr();
            auto column_ptr = column->getRawDataPtr();
            auto positions_ptr = positions->getRawDataPtr();

            /// Scalar preprocessing
            size_t pos_count = batch<scalar>::apply( result_ptr, column_ptr, positions_ptr, alignment_elements);
            std::cout << "Scalar preprocessing: " << alignment_elements << " // " << pos_count << std::endl;

            /// Vector processing
            size_t vector_count = (positions->getPopulationCount() - alignment_elements) / ps::vector_element_count();
            pos_count += batch<ps>::apply( 
                (result_ptr + pos_count), 
                column_ptr, 
                (positions_ptr + alignment_elements), 
                vector_count//,
                //alignment.getElementsUntilAlignment()
            );
            std::cout << "Vector processing: " << vector_count << " // " << pos_count << std::endl;
            /// Scalar postprocessing
            pos_count += batch<scalar>::apply( 
                result_ptr + pos_count, 
                column_ptr, 
                (positions_ptr + alignment_elements + vector_count * ps::vector_element_count()), 
                positions->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count()//,
                //alignment.getElementsUntilAlignment() + vector_count * ps::vector_element_count() 
            );
            std::cout << "Scalar postprocessing: " << column->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count() << " // " << pos_count << std::endl;

            result->setPopulationCount(pos_count);

            return result;
        }


    };

}; //namespace tuddbs

#endif //SRC_OPERATORS_PROJECT_HPP