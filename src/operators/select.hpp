#ifndef SRC_OPERATORS_SELECT_HPP
#define SRC_OPERATORS_SELECT_HPP

#include "utils/preprocessor.h"
#include "utils/AlignmentHelper.hpp"

#include <iostream>

#include <datastructure/column.hpp>

using namespace std;

namespace tuddbs{
    template<typename ProcessingStyle, template < typename ... > typename CompareOperator >
    class select {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = std::shared_ptr<col_t>;
        using const_col_ptr = std::shared_ptr<const col_t>;

        template<typename batchps>
        class batch {
            // to make it accessable by select class
            friend class select<ProcessingStyle, CompareOperator>;

            using reg_t = typename batchps::register_type;
            using mask_t = typename batchps::mask_type;

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
                    /// count the number of set bits in the mask
                    size_t count = tsl::mask_population_count<batchps>(mask);
                    /// store the positions for matched elements
                    tsl::compress_store<batchps>(mask, output_data, position_vector);
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
            typename AlignmentHelper<ps>::Alignment alignment = AlignmentHelper<ps>::getAlignment(column.get()->getRawDataPtr());


            auto result = Column<base_type>::create(column.get()->getPopulationCount(), ps::vector_size_B());

            auto result_ptr = result.get()->getRawDataPtr();
            auto column_ptr = column.get()->getRawDataPtr();


            /// Scalar preprocessing
            size_t pos_count = batch<scalar>::apply( result_ptr, column_ptr, predicate, alignment.getElementsUntilAlignment(), 0 );
            cout << "Scalar preprocessing: " << alignment.getElementsUntilAlignment() << " // " << pos_count << endl;

            /// Vector processing
            size_t vector_count = (column.get()->getPopulationCount() - alignment.getElementsUntilAlignment()) / ps::vector_element_count();
            pos_count += batch<ps>::apply( 
                (result_ptr + pos_count), 
                (column_ptr + alignment.getElementsUntilAlignment()), 
                predicate, 
                vector_count,
                alignment.getElementsUntilAlignment()
            );
            cout << "Vector processing: " << vector_count << " // " << pos_count << endl;
            /// Scalar postprocessing
            pos_count += batch<scalar>::apply( 
                result_ptr + pos_count, 
                column_ptr + alignment.getElementsUntilAlignment() + vector_count * ps::vector_element_count(), 
                predicate, 
                column.get()->getPopulationCount() - alignment.getElementsUntilAlignment() - vector_count * ps::vector_element_count(),
                alignment.getElementsUntilAlignment() + vector_count * ps::vector_element_count() 
            );
            cout << "Scalar postprocessing: " << column.get()->getPopulationCount() - alignment.getElementsUntilAlignment() - vector_count * ps::vector_element_count() << " // " << pos_count << endl;

            result.get()->setPopulationCount(pos_count);

            return result;
        }


    };

}; //namespace tuddbs



#endif //SRC_OPERATORS_SELECT_HPP