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
    
    template< typename ps , template <typename ...> typename Operator>
    class aggregate{
        using base_t = typename ps::base_type;
        using reg_t = typename ps::register_type;

        public:
        struct State{
            base_t result;
            base_t* data_ptr;   // anstatt das "const" zu entfernen vllt ne bool condition, oder vllt eine andere alternative?
            size_t count;
            bool value_there = false;   // Kann auch manchmal nur flush aufgerufen werden? wenn nein: ist das bool nicht notwendig
        };

        void operator()(State& myState){
            if constexpr(std::is_same_v<ps, typename tsl::simd<typename ps::base_type, tsl::scalar>>){
                flush(myState);
                myState.value_there = true;
                return;
            }
            size_t element_count = ps::vector_element_count();

            const base_t *end = myState.data_ptr + myState.count;

            while(myState.data_ptr <= (end - element_count)){
                reg_t vec = tsl::loadu< ps >(myState.data_ptr);

                myState.result = Operator< ps, tsl::workaround>::apply(vec);
                myState.data_ptr += element_count - 1;
                *myState.data_ptr = myState.result;
            }
            myState.value_there = true;
            
        };

        static void flush(State& myState){
            flush_helper<ps, Operator>::flush(myState);
        };

        private:
        template<typename ps_helper, template<typename...> typename Op>
        struct flush_helper{
            static void flush(typename aggregate<ps_helper, Op>::State& myState){
                static_assert(true, "No flush implementation for this operator");
            }
        };

        template<typename ps_helper>
        struct flush_helper<ps_helper, tsl::functors::hor>{
            static void flush(typename aggregate<ps_helper, tsl::functors::hor>::State& myState){
                const base_t *end = myState.data_ptr + myState.count;
                base_t temp = 0;
                while(myState.data_ptr <= end){
                    temp |= *myState.data_ptr++;
                }
                (myState.value_there)? myState.result |= temp : myState.result = temp;
            }
        };
  };
};//namespace tuddbs

#endif//SRC_OPERATORS_AGGREGATE_HPP