// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Copyright (c) 2022 SimdOperators Team.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, version 3.
 
   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.
 
   You should have received a copy of the GNU General Public License 
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
// ------------------------------------------------------------------- //
#ifndef SRC_SIMDOPERATORS_OPERATORS_SELECT_HPP
#define SRC_SIMDOPERATORS_OPERATORS_SELECT_HPP

#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/utils/constexpr/MemberDetector.h>

namespace tuddbs{

    template<typename ProcessingStyle, template<typename ...> typename CompareOperator>
    class select_core {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;
        using mask_t = typename ps::mask_type;
        using imask_t = typename ps::imask_type;


        public:
        /// Used for MetaOperator to determine if operator has a state
        constexpr static bool is_stateful = true;

        constexpr static bool is_available = detector::has_static_method_apply_v<tsl::functors::set1<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::custom_sequence<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::load<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::to_integral<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::imask_population_count<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::compress_store<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::add<ps, tsl::workaround>>;

        struct state {
            size_t pos_idx;
            state() : pos_idx(0) {}
        };

        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        static void 
        apply(base_type * out1, size_t& out1_count, const base_type * in1, size_t in1_element_count, state& state, base_type predicate1){
            out1_count = 0;

            if(in1_element_count == 0){
                return;
            }


            /// [predicate, ...]
            reg_t const predicate_vector = tsl::set1<ps>(predicate1);
            /// [element_count, ...]
            reg_t const increment_vector = tsl::set1<ps>(ps::vector_element_count());
            /// [0, 1, 2, 3, ...]
            // reg_t position_vector = tsl::sequence<ps>(); // TODO: check if sequence works correct
            reg_t position_vector = tsl::custom_sequence<ps>(state.pos_idx, 1); 


            size_t vector_count = in1_element_count / ps::vector_element_count();
            for (size_t i = 0; i < vector_count; ++i) {
                /// load data into vector register
                reg_t data_vector = tsl::load<ps>(in1);
                /// compare data with predicate, resulting in a bit mask
                mask_t mask = CompareOperator<ps, tsl::workaround>::apply(data_vector, predicate_vector);
                imask_t imask = tsl::to_integral<ps>(mask);
                /// count the number of set bits in the mask
                size_t count = tsl::mask_population_count<ps>(imask);
                /// store the positions for matched elements
                tsl::compress_store<ps>(imask, out1, position_vector);
                /// increment the position vector
                position_vector = tsl::add<ps>(position_vector, increment_vector);
                /// increment the output data pointer
                out1 += count;
                /// increment the overall count
                out1_count += count;
                /// increment the input data pointer
                in1 += ps::vector_element_count();
            }
            state.pos_idx += in1_element_count;
        }
    };


    template<typename ProcessingStyle, template < typename ... > typename CompareOperator >
    class select {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        public:

            template<typename batchps>
            class batch {
                // to make it accessable by select class
                friend class select<ProcessingStyle, CompareOperator>;

                using reg_t = typename batchps::register_type;
                using mask_t = typename batchps::mask_type;
                using imask_t = typename batchps::imask_type;

                DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
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

        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr apply(const_col_ptr column, const base_type& predicate){
            
            /// Get the alignment of the column
            typename AlignmentHelper<ps>::Alignment alignment = AlignmentHelper<ps>::getAlignment(column->getRawDataPtr());
            size_t alignment_elements;
            if (column->getPopulationCount() < alignment.getElementsUntilAlignment()) {
                alignment_elements = column->getPopulationCount();
            } else {
                alignment_elements = alignment.getElementsUntilAlignment();
            }


            auto result = new Column<base_type>(column->getPopulationCount(), ps::vector_size_B());

            auto result_ptr = result->getRawDataPtr();
            auto column_ptr = column->getRawDataPtr();


            /// Scalar preprocessing
            size_t pos_count = batch<scalar>::apply( result_ptr, column_ptr, predicate, alignment_elements, 0 );
            // std::cout << "Scalar preprocessing: " << alignment_elements << " // " << pos_count << std::endl;

            /// Vector processing
            size_t vector_count = (column->getPopulationCount() - alignment_elements) / ps::vector_element_count();
            pos_count += batch<ps>::apply( 
                (result_ptr + pos_count), 
                (column_ptr + alignment_elements), 
                predicate, 
                vector_count,
                alignment_elements
            );
            // std::cout << "Vector processing: " << vector_count << " // " << pos_count << std::endl;
            /// Scalar postprocessing
            pos_count += batch<scalar>::apply( 
                result_ptr + pos_count, 
                column_ptr + alignment_elements + vector_count * ps::vector_element_count(), 
                predicate, 
                column->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count(),
                alignment_elements + vector_count * ps::vector_element_count() 
            );
            // std::cout << "Scalar postprocessing: " << column->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count() << " // " << pos_count << std::endl;

            result->setPopulationCount(pos_count);

            return result;
        }


    };

}; //namespace tuddbs



#endif //SRC_SIMDOPERATORS_OPERATORS_SELECT_HPP
