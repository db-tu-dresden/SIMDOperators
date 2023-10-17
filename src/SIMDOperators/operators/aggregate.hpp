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

#include "/home/dertuchi/work/TSL/generated_tsl/generator_output/include/tslintrin.hpp"
#include <cassert>

namespace tuddbs {
    
    template< typename ps , template <typename ...> typename Op_h, template <typename ...> typename Op>
    class aggregate{
        using base_t = typename ps::base_type;
        using reg_t = typename ps::register_type;

        public:
        struct State{
            base_t result;
            base_t const* data_ptr;
            size_t count;
            reg_t temp;
            const bool fit_in_reg = (count >= ps::vector_element_count()); // Wie gehe ich damit um, wenn nicht genug elemente vorhanden sind um sie zu laden.
            State(base_t const* ptr, size_t cnt) : data_ptr(ptr), count(cnt), temp(tsl::loadu<ps>(ptr)){
                if(fit_in_reg){
                    data_ptr += ps::vector_element_count();
                    count -= ps::vector_element_count();
                }
            }
        };

        void operator()(State& myState){
            size_t element_count = ps::vector_element_count();
            const base_t *end = myState.data_ptr + myState.count;

            while(myState.data_ptr <= (end - element_count)){
                reg_t vec = tsl::loadu< ps >(myState.data_ptr);
                myState.temp = Op< ps, tsl::workaround>::apply(myState.temp, vec);
                myState.data_ptr += element_count;
            }
        };

        static void flush(State& myState){
            flush_helper<ps, Op_h, Op>::flush(myState);
        };

        private:
        template<typename ps_helper, template<typename...> typename Op_horizontal, template<typename...> typename Op_normal>
        struct flush_helper{
            static void flush(typename aggregate<ps_helper, Op_horizontal, Op_normal>::State& myState){
                using scalar_t = typename tsl::simd<typename ps::base_type, tsl::scalar>;
                using scalar_reg_t = typename scalar_t::register_type;

                const base_t* end = myState.data_ptr + myState.count;
                base_t result;

                if(myState.fit_in_reg){
                    result = Op_horizontal<ps, tsl::workaround>::apply(myState.temp);
                }else{
                    result = *myState.data_ptr++;
                }

                while(myState.data_ptr < end){
                    result = Op_normal<scalar_t, tsl::workaround>::apply((scalar_reg_t)result, (scalar_reg_t)*myState.data_ptr++);
                }
                myState.result = result;
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_AGGREGATE_HPP