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

#include <new>
#include <utility>
#include <cstring>
#include <utils/types.hpp>
#include <memory>
#include <cassert>

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

    template<typename base_type>
    class Column {
      private:
        /// The number of elements in this column.
        size_t element_count;
        /// The alignment of the data array (in bytes).
        size_t alignment;
        /// The offset of the data array (in bytes).
        size_t offset;
        /// The data array of this column.
        // std::shared_ptr<base_type[], std::default_delete<base_type[]>()> data;
        std::shared_ptr<base_type[]> data;
      public:

        // ========== Constructors & Destructors ================================================== //

        /// Default constructor.
        Column()
            : element_count{0},
              alignment{0},
              offset{0},
              data{nullptr}
        {}

        /// Constructor.
        Column(size_t element_count, size_t alignment=sizeof(base_type))
            : element_count{element_count},
              alignment{alignment},
              offset{0},
              data{new (std::align_val_t(alignment)) base_type[element_count]}
        {
          assert(alignment >= sizeof(base_type) && "Alignment must be at least the size of the base type.");
          assert(alignment % sizeof(base_type) == 0 && "Alignment must be a multiple of the size of the base type.");
        }

        /// Copy constructor.
        Column(const Column & other)
            : element_count{other.element_count},
              alignment{other.alignment},
              offset{other.offset},
              data{new (std::align_val_t(alignment)) base_type[element_count]} {
          std::memcpy(data.get(), other.data.get(), element_count*sizeof(base_type));
        }

        /// Move constructor.
        Column(Column && other)
            : element_count{std::exchange(other.element_count, 0)},
              alignment{std::exchange(other.alignment, 0)},
              offset{std::exchange(other.offset, 0)},
              data{std::exchange(other.data, nullptr)}
        {}

        /// Copy assignment operator.
        Column & operator=(const Column & other) {
          if(this != &other) {
            element_count = other.element_count;
            alignment = other.alignment;
            offset = other.offset;
            data = std::shared_ptr<base_type[]>(new (std::align_val_t(alignment)) base_type[element_count]);
            std::memcpy(data.get(), other.data.get(), element_count*sizeof(base_type));
          }
          return *this;
        }

        /// Move assignment operator.
        Column & operator=(Column && other) {
          if(this != &other) {
            element_count = std::exchange(other.element_count, 0);
            alignment = std::exchange(other.alignment, 0);
            offset = std::exchange(other.offset, 0);
            data = std::exchange(other.data, nullptr);
          }
          return *this;
        }

        /// Destructor.
        ~Column() = default;


        template<typename ... TArgs>
        static std::shared_ptr<Column<base_type>> create(TArgs ... args) {
          return std::make_shared<Column<base_type>>(args...);
        }

        // template<typename ... TArgs>
        // static create<


        // ========== Getter ======================================================================= //

        std::shared_ptr<const base_type[]> getData() const {
          return data;
        }

        const base_type * getRawDataPtr() const {
          return data.get();
        }
        
        std::shared_ptr<base_type[]> getData() {
          return data;
        }

        size_t getElementCount() const {
          return element_count;
        }

        size_t getAlignment() const {
          return alignment;
        }

        
        // ========== Accessors ==================================================================== //

        /// Returns a const reference to the element at the given index.
        const base_type & operator[](size_t index) const {
          return data[index + offset];
        }     





        /// Returns a column pointing into original column with the given offset (start_index) and length.
        std::shared_ptr<Column<base_type>> chunk(size_t start_index, size_t length){
          std::shared_ptr<Column<base_type>> chunk(new Column<base_type>());
          chunk->element_count = length;

          chunk->alignment = alignment;
          chunk->data = std::shared_ptr<base_type[]>(this->data, this->data.get() + start_index);

          return chunk;
        }


    };

};
#endif//SIMDOPERATORS_DBOPS_DATASTRUCTURE_INCLUDE_COLUMN_HPP
