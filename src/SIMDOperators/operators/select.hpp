#ifndef SRC_OPERATORS_SELECT_HPP
#define SRC_OPERATORS_SELECT_HPP

#include <iostream>

#include "SIMDOperators/utils/preprocessor.h"
#include "SIMDOperators/utils/AlignmentHelper.hpp"
#include <SIMDOperators/datastructure/column.hpp>

namespace tuddbs{
    template<typename ProcessingStyle, template < typename ... > typename CompareOperator >
    class select {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        template<typename batchps>
        class batch {
            // to make it accessable by select class
            friend class select<ProcessingStyle, CompareOperator>;

            using reg_t = typename batchps::register_type;
            using mask_t = typename batchps::mask_type;
            using imask_t = typename batchps::imask_type;

            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            static size_t apply(base_type * result, const base_type * column, base_type predicate, const size_t& vector_count, size_t start_index = 0){
                if(vector_count == 0){
                    return 0;
                }
                /// [predicate, ...]
                reg_t const predicate_vector = tsl::set1<batchps>(predicate);
                /// [element_count, ...]
                reg_t const increment_vector = tsl::set1<batchps>(batchps::vector_element_count());
                /// [0, 1, 2, 3, ...]
                // reg_t position_vector = tsl::sequence<batchps>(); // TODO: check if sequence works correct
                reg_t position_vector = tsl::custom_sequence<batchps>(start_index, 1); 

                /// output data pointer
                // base_type * output_data = result.get()->getRawDataPtr();
                base_type * output_data = result;
                /// input data pointer
                // base_type const * input_data = column.get()->getRawDataPtr();
                base_type const * input_data = column;

                size_t overall_count = 0;

                for (size_t i = 0; i < vector_count; ++i) {
                    /// load data into vector register
                    reg_t data_vector = tsl::load<batchps>(input_data);
                    /// compare data with predicate, resulting in a bit mask
                    mask_t mask = CompareOperator<batchps, tsl::workaround>::apply(data_vector, predicate_vector);
                    imask_t imask = tsl::to_integral<batchps>(mask);
                    /// count the number of set bits in the mask
                    size_t count = tsl::mask_population_count<batchps>(imask);
                    /// store the positions for matched elements
                    tsl::compress_store<batchps>(imask, output_data, position_vector);
                    /// increment the position vector
                    position_vector = tsl::add<batchps>(position_vector, increment_vector);
                    /// increment the output data pointer
                    output_data += count;
                    /// increment the overall count
                    overall_count += count;
                    /// increment the input data pointer
                    input_data += batchps::vector_element_count();
                }
                return overall_count;
            }
        };

    public:

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr apply(const_col_ptr column, const base_type& predicate){
            
            /// Get the alignment of the column
            typename AlignmentHelper<ps>::Alignment alignment = AlignmentHelper<ps>::getAlignment(column->getRawDataPtr());
            size_t alignment_elements;
            if (column->getPopulationCount() < alignment.getElementsUntilAlignment()) {
                alignment_elements = column->getPopulationCount();
            } else {
                alignment_elements = alignment.getElementsUntilAlignment();
            }


            auto result = Column<base_type>::create(column->getPopulationCount(), ps::vector_size_B());

            auto result_ptr = result->getRawDataPtr();
            auto column_ptr = column->getRawDataPtr();


            /// Scalar preprocessing
            size_t pos_count = batch<scalar>::apply( result_ptr, column_ptr, predicate, alignment_elements, 0 );
            std::cout << "Scalar preprocessing: " << alignment_elements << " // " << pos_count << std::endl;

            /// Vector processing
            size_t vector_count = (column->getPopulationCount() - alignment_elements) / ps::vector_element_count();
            pos_count += batch<ps>::apply( 
                (result_ptr + pos_count), 
                (column_ptr + alignment_elements), 
                predicate, 
                vector_count,
                alignment_elements
            );
            std::cout << "Vector processing: " << vector_count << " // " << pos_count << std::endl;
            /// Scalar postprocessing
            pos_count += batch<scalar>::apply( 
                result_ptr + pos_count, 
                column_ptr + alignment_elements + vector_count * ps::vector_element_count(), 
                predicate, 
                column->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count(),
                alignment_elements + vector_count * ps::vector_element_count() 
            );
            std::cout << "Scalar postprocessing: " << column->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count() << " // " << pos_count << std::endl;

            result->setPopulationCount(pos_count);

            return result;
        }


    };

}; //namespace tuddbs



#endif //SRC_OPERATORS_SELECT_HPP
