// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Copyright (c) 2022 Johannes Pietrzyk.
   
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

/*
 * @file column.hpp
 * @author jpietrzyk
 * @date 14.07.22
 * @brief A brief description.
 *
 * @details A detailed description.
 */

#ifndef SIMDOPERATORS_DBOPS_DATASTRUCTURE_INCLUDE_COLUMN_HPP
#define SIMDOPERATORS_DBOPS_DATASTRUCTURE_INCLUDE_COLUMN_HPP

#include <utility>
#include <cstring>
#include <types.hpp>

namespace tuddbs{
template<Arithmetic T>
struct column_t {
 public:
  using base_type = T;
 private:
  size_t m_element_count;
  size_t m_alignment;
  T * m_raw_data;
 public:
  T const * data() const {
    return m_raw_data;
  }
  T* data() {
    return m_raw_data;
  }
  size_t element_count() const {
    return m_element_count;
  }
  size_t alignment() const {
    return m_alignment;
  }
 public:
  column_t()
      : m_element_count{0},
        m_alignment{0},
        m_raw_data{nullptr}
  {}
  explicit column_t(size_t p_element_count, size_t p_alignment=1)
      : m_element_count{p_element_count},
        m_alignment{p_alignment},
        m_raw_data{reinterpret_cast<T*>(std::aligned_alloc(p_alignment, p_element_count*sizeof(T)))}
  {}
  ~column_t() {
    if(m_raw_data != nullptr) {
      free(m_raw_data);
    }
  }
  column_t(column_t const & other)
      : m_element_count{other.m_element_count},
        m_alignment{other.m_alignment},
        m_raw_data{reinterpret_cast<T*>(std::aligned_alloc(m_alignment, m_element_count*sizeof(T)))} {
    std::copy(other.m_raw_data, other.m_raw_data + m_element_count, m_raw_data);
  }
  column_t & operator=(column_t const & other) {
    if(this != &other) {
      m_element_count = other.m_element_count;
      m_alignment = other.m_alignment;
      m_raw_data = reinterpret_cast<T*>(std::aligned_alloc(m_alignment, m_element_count*sizeof(T)));
      std::copy(other.m_raw_data, other.m_raw_data + m_element_count, m_raw_data);
    }
    return *this;
  }
  column_t(column_t && other)
      : m_element_count{std::exchange(other.m_element_count, 0)},
        m_alignment{std::exchange(other.m_alignment, 0)},
        m_raw_data{std::exchange(other.m_raw_data, nullptr)}
  {}
  column_t & operator=(column_t && other) {
    if(this != &other) {
      m_element_count = std::exchange(other.m_element, 0);
      m_alignment = std::exchange(other.m_alignment, 0);
      m_raw_data = std::exchange(other.m_raw_data, nullptr);
    }
    return *this;
  }
};
}
#endif//SIMDOPERATORS_DBOPS_DATASTRUCTURE_INCLUDE_COLUMN_HPP
