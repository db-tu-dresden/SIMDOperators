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

#ifndef SRC_SIMDOPERATORS_UTILS_BASICSTASH_HPP
#define SRC_SIMDOPERATORS_UTILS_BASICSTASH_HPP

#include <cstddef>

namespace tuddbs {
    template<typename ProcessingStyle, size_t BatchSizeInBytes>
    class basic_stash_t {
        using data_ptr_t = typename ProcessingStyle::base_type const *;
        using result_ptr_t = typename ProcessingStyle::base_type *;
        private:
            data_ptr_t m_data_ptr;
            result_ptr_t m_result_ptr;
        public:
            void data_ptr(data_ptr_t p_data_ptr) {
                m_data_ptr = p_data_ptr;
            }
            data_ptr_t data_ptr() const {
                return m_data_ptr;
            }
            void advance() {
                m_data_ptr += BatchSizeInBytes / sizeof(typename ProcessingStyle::base_type);
            }
            void result_ptr(result_ptr_t p_result_ptr) {
                m_result_ptr = p_result_ptr;
            }
            result_ptr_t result_ptr() const {
                return m_result_ptr;
            }
            size_t element_count() const {
                return BatchSizeInBytes / sizeof(typename ProcessingStyle::base_type);
            }
        public: 
            explicit basic_stash_t(data_ptr_t p_data_ptr, result_ptr_t p_result_ptr) 
            : m_data_ptr(p_data_ptr), 
              m_result_ptr(p_result_ptr) 
            {}

    };

}; // namespace tuddbs

#endif // SRC_SIMDOPERATORS_UTILS_BASICSTASH_HPP