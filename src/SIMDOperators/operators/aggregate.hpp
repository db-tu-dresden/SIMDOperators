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
#ifndef SRC_OPERATORS_AGGREGATE_HPP
#define SRC_OPERATORS_AGGREGATE_HPP

#include <tslintrin.hpp>
#include <cassert>

namespace tuddbs {
    template< typename ps , template <typename ...> typename Op, template <typename ...> typename Op_h>
    class aggregate{
        using base_t = typename ps::base_type;
        using reg_t = typename ps::register_type;

        public:
        struct State{
            base_t result;
            base_t const* data_ptr;
            size_t count;
            reg_t temp;
            const bool fit_in_reg = (count >= ps::vector_element_count());
            State(base_t const* ptr, size_t cnt) : data_ptr(ptr), count(cnt), temp(tsl::loadu<ps>(ptr)){
                /*
                    fit_in_reg is needed to check if the initialized temp register can be used or not.
                    If the batch_size is smaller than the vector_element_count we can not load data into the register.
                    This also means we have to handle the flush differently.
                */ 
                if(fit_in_reg){
                    data_ptr += ps::vector_element_count();
                    count -= ps::vector_element_count();
                }
            }
        };

        void operator()(State& myState){
            size_t element_count = ps::vector_element_count();
            const base_t *end = myState.data_ptr + myState.count;

            // Apply the Operator on temp and data and save the result in temp.
            while(myState.data_ptr <= (end - element_count)){
                reg_t vec = tsl::loadu< ps >(myState.data_ptr);
                myState.temp = Op< ps, tsl::workaround>::apply(myState.temp, vec);
                myState.data_ptr += element_count;
            }
        };

        static void flush(State& myState){
            using scalar_t = typename tsl::simd<typename ps::base_type, tsl::scalar>;
            using scalar_reg_t = typename scalar_t::register_type;

            const base_t* end = myState.data_ptr + myState.count;
            base_t result;

            /*
                if temp is valid (fit_in_reg == true) we use the horizontal Operation to calculate the aggregation result.
                Else we cant use the temp register and have to initialize result with the first value of our dataptr.
            */
            if(myState.fit_in_reg){
                result = Op_h<ps, tsl::workaround>::apply(myState.temp);
            }else{
                result = *myState.data_ptr++;
            }

            // Remaining elements are calculated together scalar whise with result.
            while(myState.data_ptr < end){
                result = Op<scalar_t, tsl::workaround>::apply((scalar_reg_t)result, (scalar_reg_t)*myState.data_ptr++);
            }
            myState.result = result;
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_AGGREGATE_HPP