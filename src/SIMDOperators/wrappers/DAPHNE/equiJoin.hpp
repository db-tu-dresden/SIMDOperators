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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_EQUIJOIN_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_EQUIJOIN_HPP
#include "SIMDOperators/utils/BasicStash.hpp"
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/operators/naturalEquiJoin.hpp>

namespace tuddbs{
    template<typename ProcessingStyle, class DataStructure>
    class daphne_equi_join {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
            static std::tuple<col_ptr, col_ptr> apply(const_col_ptr lhs, const_col_ptr rhs){
                
                /// Get the alignment of the columns
                typename AlignmentHelper<ps>::Alignment alignment_lhs = AlignmentHelper<ps>::getAlignment(lhs->getRawDataPtr());
                size_t alignment_elements_lhs;
                if (lhs->getPopulationCount() < alignment_lhs.getElementsUntilAlignment()) {
                    alignment_elements_lhs = lhs->getPopulationCount();
                } else {
                    alignment_elements_lhs = alignment_lhs.getElementsUntilAlignment();
                }

                typename AlignmentHelper<ps>::Alignment alignment_rhs = AlignmentHelper<ps>::getAlignment(rhs->getRawDataPtr());
                size_t alignment_elements_rhs;
                if (rhs->getPopulationCount() < alignment_rhs.getElementsUntilAlignment()) {
                    alignment_elements_rhs = rhs->getPopulationCount();
                } else {
                    alignment_elements_rhs = alignment_rhs.getElementsUntilAlignment();
                }

                const size_t inBuildDataCount = lhs->getPopulationCount();
                const size_t inProbeDataCount = rhs->getPopulationCount();

                const size_t outCountEstimate = 0;

                const size_t outCount = bool(outCountEstimate)
                                    // use given estimate
                                    ? (outCountEstimate)
                                    // use pessimistic estimate
                                    : (inProbeDataCount);

                base_type * lhs_ptr = const_cast<base_type *>(lhs->getRawDataPtr());
                base_type * rhs_ptr = const_cast<base_type *>(rhs->getRawDataPtr());

                DataStructure hs( inBuildDataCount );

                auto result_lhs = Column<base_type>::create(outCount, ps::vector_size_B());
                auto result_rhs = Column<base_type>::create(outCount, ps::vector_size_B());

                auto result_lhs_ptr = result_lhs->getRawDataPtr();
                auto result_rhs_ptr = result_rhs->getRawDataPtr();

                /// Build phase
                naturalEquiJoin<scalar, DataStructure>::equi_join_build_batch::apply( lhs_ptr, alignment_elements_lhs, 0, hs );
                size_t vector_count_build = (lhs->getPopulationCount() - alignment_elements_lhs) / ps::vector_element_count();
                naturalEquiJoin<ProcessingStyle, DataStructure>::equi_join_build_batch::apply( lhs_ptr, vector_count_build, alignment_elements_lhs, hs );
                
                naturalEquiJoin<scalar, DataStructure>::equi_join_build_batch::apply( lhs_ptr, lhs->getPopulationCount() - alignment_elements_lhs - vector_count_build * ps::vector_element_count(), alignment_elements_lhs + vector_count_build * ps::vector_element_count(), hs );
                
                /// Probe phase
                size_t result_count = naturalEquiJoin<scalar, DataStructure>::equi_join_probe_batch::apply( rhs_ptr, alignment_elements_rhs, result_lhs_ptr, result_rhs_ptr, 0, hs );
                
                size_t vector_count_probe = (rhs->getPopulationCount() - alignment_elements_rhs) / ps::vector_element_count();
                result_count += naturalEquiJoin<ProcessingStyle, DataStructure>::equi_join_probe_batch::apply( rhs_ptr , vector_count_probe, result_lhs_ptr, result_rhs_ptr, alignment_elements_rhs, hs );
                
                result_count += naturalEquiJoin<scalar, DataStructure>::equi_join_probe_batch::apply( rhs_ptr, rhs->getPopulationCount() - alignment_elements_rhs - vector_count_probe * ps::vector_element_count(), result_lhs_ptr, result_rhs_ptr  , alignment_elements_rhs + vector_count_probe * ps::vector_element_count(), hs );
                
                result_lhs->setPopulationCount(result_count);
                result_rhs->setPopulationCount(result_count);

                return std::make_tuple(result_lhs, result_rhs);
            }


        };

        //Define convenience function
        template<
            typename VectorExtension
        >
        std::tuple<
            Column< typename VectorExtension::base_type > *,
            Column< typename VectorExtension::base_type > *
        > 
        natural_equi_join(
        Column< typename VectorExtension::base_type > const * const p_InDataLCol,
        Column< typename VectorExtension::base_type > const * const p_InDataRCol,
        size_t const outCountEstimate = 0
        ) {
        return naturalEquiJoin<
            VectorExtension,
            vectorlib::hash_map<
                VectorExtension,
                vectorlib::multiply_mod_hash,
                vectorlib::size_policy_hash::EXPONENTIAL,
                vectorlib::scalar_key_vectorized_linear_search,
                60
            >
        >::apply(p_InDataLCol,p_InDataRCol);
        }

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_EQUIJOIN_HPP
