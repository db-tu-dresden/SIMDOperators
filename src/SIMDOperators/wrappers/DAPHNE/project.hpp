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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_PROJECT_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_PROJECT_HPP
#include "SIMDOperators/utils/BasicStash.hpp"
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/operators/project.hpp>

namespace tuddbs{
    template<typename ProcessingStyle>
    class daphne_project {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
            col_ptr operator()(const_col_ptr column, const_col_ptr positions){
                using ps = ProcessingStyle;
                using base_type = typename ps::base_type;
                using scalar = tsl::simd<base_type, tsl::scalar>;

                using col_t = Column<base_type>;
                using col_ptr = col_t *;
                using const_col_ptr = const col_t *;
                /// Get the alignment of the positions column
                typename AlignmentHelper<ps>::Alignment alignment = AlignmentHelper<ps>::getAlignment(positions->getRawDataPtr());
                size_t alignment_elements;
                if (positions->getPopulationCount() < alignment.getElementsUntilAlignment()) {
                    alignment_elements = positions->getPopulationCount();
                } else {
                    alignment_elements = alignment.getElementsUntilAlignment();
                }

                Column<base_type> * result = Column<base_type>::create(positions->getPopulationCount(), ps::vector_size_B());

                auto result_ptr = result->getRawDataPtr();
                auto column_ptr = column->getRawDataPtr();
                auto positions_ptr = positions->getRawDataPtr();

                /// Scalar preprocessing
                size_t batch_size_pre = alignment_elements * sizeof(base_type);
                tuddbs::basic_stash_t<scalar, scalar::vector_size_B()> state_pre(positions_ptr, result_ptr);
                tuddbs::project<scalar, scalar::vector_size_B()> project_pre;
                for (size_t batch = 0; batch < batch_size_pre / scalar::vector_size_B(); ++batch) {
                    project_pre(
                        state_pre, 
                        column_ptr);
                }

                /// Vector processing
                size_t vector_count = (positions->getPopulationCount() - alignment_elements) / ps::vector_element_count();
                size_t batch_size_vec = vector_count * ps::vector_element_count() * sizeof(base_type);
                tuddbs::basic_stash_t<ps, ps::vector_size_B()> state_vec(positions_ptr + alignment_elements, result_ptr + alignment_elements);
                tuddbs::project<ps, ps::vector_size_B()> project_vec;
                for (size_t batch = 0; batch < batch_size_vec / ps::vector_size_B(); ++batch) {
                    project_vec( 
                        state_vec,
                        column_ptr
                    );
                }
                
                /// Scalar postprocessing
                size_t batch_size_post = (positions->getPopulationCount() - alignment_elements - vector_count * ps::vector_element_count()) * sizeof(base_type);
                tuddbs::basic_stash_t<scalar, scalar::vector_size_B()> state_post(positions_ptr + alignment_elements + vector_count * ps::vector_element_count(), result_ptr + alignment_elements + vector_count * ps::vector_element_count());
                tuddbs::project<scalar, scalar::vector_size_B()> project_post;
                for (size_t batch = 0; batch < batch_size_post / scalar::vector_size_B(); ++batch) {
                    project_post( 
                        state_post,
                        column_ptr
                    );
                }

                result->setPopulationCount(positions->getPopulationCount());

                return result;
            }


    };

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_PROJECT_HPP
