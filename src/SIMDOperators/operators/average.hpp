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

#include "/home/tucholke/work/TSL/generated_tsl/generator_output/include/tslintrin.hpp"
#include "aggregate.hpp"
#include <cassert>

namespace tuddbs {
    template< typename ps, typename result_t = float>
    class average{
        using base_t = typename ps::base_type;
        using reg_t = typename ps::register_type;
        // using result_t = typename std::conditional<std::is_integral<base_t>::value, float, base_t>::type;

        public:
        struct State{
            result_t result;
            base_t const* data_ptr;
            size_t count;
        };

        void operator()(State& myState){
            size_t element_count = ps::vector_element_count();
            const base_t *end = myState.data_ptr + myState.count;
            
            typename tuddbs::aggregate<ps, tsl::functors::add, tsl::functors::hadd>::State state_aggregate(myState.data_ptr, myState.count);

            tuddbs::aggregate<ps, tsl::functors::add, tsl::functors::hadd>{}(state_aggregate);
            state_aggregate.count = end - state_aggregate.data_ptr;
            tuddbs::aggregate<ps, tsl::functors::add, tsl::functors::hadd>::flush(state_aggregate);

            myState.result = tsl::functors::div<typename tsl::simd<result_t, tsl::scalar>, tsl::workaround>::apply(state_aggregate.result, myState.count);
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_AVERAGE_HPP