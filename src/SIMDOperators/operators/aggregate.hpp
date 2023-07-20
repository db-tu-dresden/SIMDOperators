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
#include <tslintrin.hpp>

namespace tuddbs{
  template<typename ProcessingStyle, size_t BatchSizeInBytes, template<typename...> class AggregationOperator, template<typename...> class ReduceAggregationOperator, typename Idof = tsl::workaround>
    class aggregate {
      static_assert(BatchSizeInBytes % ProcessingStyle::vector_size_B() == 0, "BatchSizeInBytes must be a multiple of the vector size in bytes");
      public:
        using data_ptr_t            = typename ProcessingStyle::base_type const *;
        using result_t              = typename ProcessingStyle::base_type;
      public:
        
        
        /**
         * @brief 
         * @details The `intermediate_state_t` struct is defining a data structure that holds the intermediate state during the aggregation process. 
         *          It is used to store the data pointer, the number of elements to process and the current result.
         *          For the intermediate state we assume, that the element count is a multiple of the vector element count.
         */
        class intermediate_state_t {
          friend class flush_state_t;
          using iresult_t = typename ProcessingStyle::register_type;
          private:
            data_ptr_t m_data_ptr;
            iresult_t  m_result;
          public:
            void data_ptr(data_ptr_t _data_ptr) {
              m_data_ptr = _data_ptr;
            }
            data_ptr_t data_ptr() const {
              return m_data_ptr;
            }
            void advance() {
              m_data_ptr += BatchSizeInBytes / sizeof(typename ProcessingStyle::base_type);
            }
            iresult_t result() const {
              return m_result;
            }
            void result(iresult_t result) {
              m_result = result;;
            }
            
            size_t element_count() const {
              return BatchSizeInBytes / sizeof(typename ProcessingStyle::base_type);
            }
          public:
            explicit intermediate_state_t(data_ptr_t data_ptr)
            : m_data_ptr(data_ptr), 
              m_result(tsl::set1<ProcessingStyle>(0)) 
            {}
            intermediate_state_t(data_ptr_t data_ptr, iresult_t result)
            : m_data_ptr(data_ptr), 
              m_result(result) 
            {}
        };
        class flush_state_t {
          private:
            data_ptr_t m_data_ptr;
            size_t     m_element_count;
            result_t   m_result;
          public:
            result_t result() const {
              return m_result;
            }
            void result(result_t result) {
              m_result = result;
            }
            data_ptr_t data_ptr() const {
              return m_data_ptr;
            }
            size_t element_count() const {
              return m_element_count;
            }
          public:
            flush_state_t(data_ptr_t data_ptr, size_t element_count, intermediate_state_t const & intermediate_state)
            : m_data_ptr(data_ptr), 
              m_element_count(element_count), 
              m_result(ReduceAggregationOperator<ProcessingStyle, Idof>::apply(intermediate_state.result())) 
            {}
            flush_state_t(size_t element_count, intermediate_state_t const & intermediate_state)
            : m_data_ptr(intermediate_state.data_ptr()), 
              m_element_count(element_count), 
              m_result(ReduceAggregationOperator<ProcessingStyle, Idof>::apply(intermediate_state.result())) 
            {
            }
          
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
              tsl::simd<typename ProcessingStyle::base_type, tsl::scalar>
            >;
          
          auto result = state.result();
          for (size_t i = 0; i < state.element_count(); i += ps::vector_element_count()) {
            auto const data = tsl::loadu<ps>(state.data_ptr() + i);
            result = AggregationOperator<ps, Idof>::apply(result, data);
          }
          state.result(result);
        }
    };

}

#endif //SRC_SIMDOPERATORS_OPERATORS_AGGREGATE_HPP
