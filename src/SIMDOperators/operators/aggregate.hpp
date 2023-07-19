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
#ifndef SRC_SIMDOPERATORS_OPERATORS_AGGREGATE_HPP
#define SRC_SIMDOPERATORS_OPERATORS_AGGREGATE_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tuddbs{
  template<typename ProcessingStyle, template<typename> AggregationOperator, template<typename> ReduceAggregationOperator>
    class select {
      public:
        using data_ptr_t            = typename ProcessingStyle::base_type const *;
        using result_t              = typename ProcessingStyle::base_type;
      public:
        
        
        /**
         * @brief 
         * @details The `intermediate_state_t` struct is defining a data structure that holds the intermediate state during the aggregation process. 
         */
        struct intermediate_state_t {
          using iresult_t = typename ProcessingStyle::register_type;
          data_ptr_t data_ptr;
          size_t     element_count;
          iresult_t  result;
        };
        class flush_state_t {
          private:
            data_ptr_t data_ptr;
            size_t     element_count;
            result_t   result;
          public:
            flush_state_t(data_ptr_t data_ptr, size_t element_count,  intermediate_state_t const & intermediate_state)
            : data_ptr(data_ptr), 
              element_count(element_count), 
              result(ReduceAggregationOperator<ProcessingStyle>::apply(result)) 
            {}
        };
      public:
        template<typename StateT>
        void operator()(StateT & state) {
          using is_intermediate_state = std::is_same<StateT, intermediate_state_t>;
          using is_flush_state = std::is_same<StateT, flush_state_t>;
          static_assert(
            is_intermediate_state::value || is_flush_state::value,
            "StateT must be either intermediate_state_t or flush_state_t"
          );
          using ps = 
            std::conditional_t<
              is_intermediate_state::value, 
              ProcessingStyle, 
              tsl::simd<ProcessingStyle::base_type, tsl::scalar>
            >;
          
          auto result = state.result;
          for (size_t i = 0; i < state.element_count; i += ps::vector_element_count()) {
            auto const data = ps::loadu(state.data_ptr + i);
            result = AggregationOperator<ps>::apply(result, data);
          }
          state.result = result;
        }
    };

}

#endif //SRC_SIMDOPERATORS_OPERATORS_AGGREGATE_HPP
