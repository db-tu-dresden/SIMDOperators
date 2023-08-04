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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_AGGREGATE_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_AGGREGATE_HPP
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/operators/aggregate.hpp>

namespace tuddbs{
    template<typename ProcessingStyle, template<typename...> class AggregationOperator, template<typename...> class ReduceAggregationOperator>
    class daphne_aggregate {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
            col_ptr operator()(const_col_ptr column){

                auto result = new Column<base_type>(1, ps::vector_size_B());

                auto result_ptr = result->getRawDataPtr();
                auto column_ptr = column->getRawDataPtr();

                auto batch_size = ps::vector_size_B();
                auto data_size = ps::vector_size_B() * column->getPopulationCount() / ps::vector_element_count();

                using op_t = tuddbs::aggregate<ps, batch_size, AggregationOperator, ReduceAggregationOperator>;
                typename op_t::intermediate_state_t state(column_ptr);
                op_t aggregate;
                for (size_t batch = 0; batch < data_size / batch_size; ++batch) {
                    aggregate(state);
                    state.advance();
                }
                typename op_t::flush_state_t flush_state(data_size % batch_size, state);
                aggregate(flush_state);


                result->setPopulationCount(1);
                return result;
            }


    };

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_AGGREGATE_HPP
