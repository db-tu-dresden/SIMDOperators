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
#ifndef SRC_OPERATORS_AVERAGE_HPP
#define SRC_OPERATORS_AVERAGE_HPP

#include <tslintrin.hpp>
#include "aggregate.hpp"
#include <cassert>

namespace tuddbs {
    // result_type uses base_type per default to check for "using result_t". If its Integral, use float as return type. Else use base_type as result_type.
    template< typename ps, typename result_type = typename ps::base_type>
    class average{
        using base_t = typename ps::base_type;
        using reg_t = typename ps::register_type;
        using result_t = typename std::conditional<std::is_integral<result_type>::value, float, result_type>::type;

        public:
        struct State{
            result_t result;
            base_t const* data_ptr;
            size_t count;

            // needed to save results from batch calculation.
            base_t temp_aggregate = 0;
            size_t global_count = 0;
        };

        void operator()(State& myState){
            const base_t *end = myState.data_ptr + myState.count;

            // Use aggregate with add to add all batch_values together.
            typename tuddbs::aggregate<ps, tsl::functors::add, tsl::functors::hadd>::State state_aggregate(myState.data_ptr, myState.count);
            tuddbs::aggregate<ps, tsl::functors::add, tsl::functors::hadd>{}(state_aggregate);
            state_aggregate.count = end - state_aggregate.data_ptr;
            tuddbs::aggregate<ps, tsl::functors::add, tsl::functors::hadd>::flush(state_aggregate);

            // Save global element_count and calculated value
            myState.temp_aggregate += state_aggregate.result;
            myState.global_count += myState.count;
            myState.data_ptr += myState.count;
        };

        static void finalize(State& myState){
            const base_t *end = myState.data_ptr + myState.count;
            
            // add remaining elements together
            while(myState.data_ptr < end){
                myState.temp_aggregate = tsl::functors::add<typename tsl::simd<typename ps::base_type, tsl::scalar>, tsl::workaround>::apply(myState.temp_aggregate, *myState.data_ptr++);
            }
            myState.global_count += myState.count;

            // divide with global element count
            myState.result = tsl::functors::div<typename tsl::simd<result_t, tsl::scalar>, tsl::workaround>::apply((result_t)myState.temp_aggregate, (result_t)myState.global_count);
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_AVERAGE_HPP