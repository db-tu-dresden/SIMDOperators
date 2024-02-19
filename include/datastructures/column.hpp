// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Author(s): Johannes Pietrzyk.

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
/**
 * @file column.hpp
 * @brief Defines the InMemoryColumn class for storing columnar data in
 * memory.
 */

#ifndef SIMDOPS_INCLUDE_DATASTRUCTURES_COLUMN_HPP
#define SIMDOPS_INCLUDE_DATASTRUCTURES_COLUMN_HPP
#include "iterable.hpp"

namespace tuddbs {
  template <tsl::TSLArithmetic T, class Allocator = T *(*)(size_t), class Deleter = void (*)(T *)>
  /**
   * @class InMemoryColumn
   * @brief Represents a column of data stored in memory.
   * @tparam T The type of data stored in the column.
   */
  class InMemoryColumn {
   public:
    using base_type = T;                        /**< The base type of the column data. */
    using pointer_type = decltype(Allocator()); /**< Pointer type for accessing the column data. */
    using const_pointer_type = T *const;        /**< Const pointer type for accessing the column data. */

   private:
    Allocator m_allocator = [](size_t i) {
      return reinterpret_cast<T *>(new T[i]);
    }; /**< Allocator function for allocating memory for the column data. */
    Deleter m_deleter = [](T *ptr) {
      delete[] ptr;
    }; /**< Deleter function for deallocating memory for the column data. */

   protected:
    T *m_data; /**< Pointer to the start of the column data. */
    T *m_end;  /**< Pointer to the end of the column data. */

   private:
    size_t m_count;          /**< Number of elements in the column. */
    size_t m_size;           /**< Size of the column data in bytes. */
    bool m_owns_data = true; /**< Flag indicating whether the column owns the data or not. */

   private:
    /**
     * @brief Checks if the column should remove the memory it owns.
     * @return True if the column should remove the memory, false otherwise.
     */
    bool should_remove_memory() const noexcept {
      if ((m_owns_data) && (m_data != nullptr)) {
        return true;
      }
      return false;
    }

   public:
    /**
     * @brief Constructs an InMemoryColumn with the specified number of
     * elements.
     * @param count The number of elements in the column.
     */
    explicit InMemoryColumn(size_t count) noexcept
      : m_data(m_allocator(count)), m_end(m_data + count), m_count(count), m_size(count * sizeof(T)) {}

    /**
     * @brief Constructs an InMemoryColumn with the specified number of
     * elements, allocator, and deleter.
     * @param count The number of elements in the column.
     * @param allocator The allocator function for allocating memory.
     * @param deleter The deleter function for deallocating memory.
     */
    explicit InMemoryColumn(size_t count, Allocator &&allocator, Deleter &&deleter) noexcept
      : m_allocator(allocator),
        m_deleter(deleter),
        m_data(m_allocator(count)),
        m_end(m_data + count),
        m_count(count),
        m_size(count * sizeof(T)) {}

    /**
     * @brief Constructs an InMemoryColumn with the specified pointer to data,
     * number of elements, and deleter.
     * @param data The pointer to the data.
     * @param count The number of elements in the column.
     * @param deleter The deleter function for deallocating memory.
     */
    explicit InMemoryColumn(pointer_type &&data, size_t count, Deleter &&deleter) noexcept
      : m_deleter(deleter),
        m_data(std::exchange(data, nullptr)),
        m_end(m_data + count),
        m_count(count),
        m_size(count * sizeof(T)) {}

    /**
     * @brief Constructs an InMemoryColumn with the specified pointer to data
     * and number of elements.
     * @param data The pointer to the data.
     * @param count The number of elements in the column.
     */
    explicit InMemoryColumn(pointer_type data, size_t count) noexcept
      : m_data(data), m_end(m_data + count), m_count(count), m_size(count * sizeof(T)), m_owns_data(false) {}

    /**
     * @brief Deleted copy constructor.
     */
    InMemoryColumn(InMemoryColumn const &) = delete;

    /**
     * @brief Move constructor for InMemoryColumn.
     * @param other The InMemoryColumn to move from.
     */
    InMemoryColumn(InMemoryColumn &&other) noexcept
      : m_allocator(std::exchange(other.m_allocator, nullptr)),
        m_deleter(std::exchange(other.m_deleter, nullptr)),
        m_data(std::exchange(other.m_data, nullptr)),
        m_end(std::exchange(other.m_end, nullptr)),
        m_count(std::exchange(other.m_count, 0)),
        m_size(std::exchange(other.m_size, 0)),
        m_owns_data(std::exchange(other.m_owns_data, false)) {}

    /**
     * @brief Deleted copy assignment operator.
     */
    InMemoryColumn &operator=(InMemoryColumn const &) = delete;

    /**
     * @brief Move assignment operator for InMemoryColumn.
     * @param other The InMemoryColumn to move from.
     * @return Reference to the moved InMemoryColumn.
     */
    InMemoryColumn &operator=(InMemoryColumn &&other) noexcept {
      if (this != &other) {
        if (should_remove_memory()) {
          m_deleter(m_data);
        }
        m_allocator = std::exchange(other.m_allocator, nullptr);
        m_deleter = std::exchange(other.m_deleter, nullptr);
        m_data = std::exchange(other.m_data, nullptr);
        m_end = std::exchange(other.m_end, nullptr);
        m_count = std::exchange(other.m_count, 0);
        m_size = std::exchange(other.m_size, 0);
        m_owns_data = std::exchange(other.m_owns_data, false);
      }
      return *this;
    }

    /**
     * @brief Destructor for InMemoryColumn.
     */
    virtual ~InMemoryColumn() noexcept {
      if (should_remove_memory()) {
        m_deleter(m_data);
      }
    }

   protected:
    /**
     * @brief Checks if the index is valid.
     * @tparam IncludeBoundary Flag indicating whether to include the boundary
     * in the check.
     * @param idx The index to check.
     * @return True if the index is valid, false otherwise.
     */
    template <bool IncludeBoundary = false>
    bool valid_index(Integral auto idx) const noexcept {
      if constexpr (std::is_unsigned_v<decltype(idx)>) {
        if constexpr (IncludeBoundary) {
          if (idx <= m_count) [[likely]] {
            return true;
          } else [[unlikely]] {
            return false;
          }
        } else {
          if (idx < m_count) [[likely]] {
            return true;
          } else [[unlikely]] {
            return false;
          }
        }
      } else {
        if constexpr (IncludeBoundary) {
          if ((idx >= 0) && (idx <= m_count)) [[likely]] {
            return true;
          } else [[unlikely]] {
            return false;
          }
        } else {
          if ((idx >= 0) && (idx < m_count)) [[likely]] {
            return true;
          } else [[unlikely]] {
            return false;
          }
        }
      }
    }

   public:
    /**
     * @brief Sets the value at the specified index.
     * @param value The value to set.
     * @param idx The index to set the value at.
     * @throws std::out_of_range Throws if index is out of range
     */
    void set_value(T value, size_t idx) {
      if (valid_index(idx)) [[likely]] {
        m_data[idx] = value;
      } else [[unlikely]] {
        throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
      }
    }

    /**
     * @brief Gets the value at the specified index.
     * @param idx The index to get the value from.
     * @return The value at the specified index.
     * @throws std::out_of_range Throws if index is out of range
     */
    T get_value(size_t idx = 0) const {
      if (valid_index(idx)) [[likely]] {
        return m_data[idx];
      } else [[unlikely]] {
        throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
      }
    }

    /**
     * @brief Gets the size of the column data in bytes.
     * @return The size of the column data in bytes.
     */
    auto size() const noexcept -> size_t { return m_size; }

    /**
     * @brief Gets the number of elements in the column.
     * @return The number of elements in the column.
     */
    auto count() const noexcept -> size_t { return m_count; }

    auto allocator() const noexcept -> Allocator { return m_allocator; }
    auto deleter() const noexcept -> Deleter { return m_deleter; }

   public:
    /**
     * @brief Iterator class for iterating over the elements of the column.
     * @tparam Const Flag indicating whether the iterator is const or not.
     */
    template <bool Const>
    class iterator {
     public:
      using base_type = std::conditional_t<Const, T const, T>; /**< The base type of the iterator. */
      using pointer_type = base_type *;                        /**< Pointer type for accessing the iterator value. */
      using deref_type = std::conditional_t<Const, T const &, T &>;  /**< Type for dereferencing the iterator. */
      using void_type = std::conditional_t<Const, void const, void>; /**< Void type for casting the iterator to a
                                                                        void pointer. */

     protected:
      pointer_type m_data; /**< Pointer to the current element of the iterator. */

     public:
      /**
       * @brief Constructs an iterator with the specified data pointer.
       * @param data The data pointer.
       */
      explicit iterator(pointer_type data) noexcept : m_data(data) {}

      /**
       * @brief Copy constructor for the iterator.
       * @param other The iterator to copy from.
       */
      iterator(iterator const &other) noexcept : m_data(other.m_data){};

      /**
       * @brief Move constructor for the iterator.
       * @param other The iterator to move from.
       */
      iterator(iterator &&other) noexcept : m_data(std::exchange(other.m_data, nullptr)) {}

      /**
       * @brief Copy assignment operator for the iterator.
       * @param other The iterator to copy from.
       * @return Reference to the copied iterator.
       */
      iterator &operator=(iterator const &other) noexcept {
        if (this != &other) {
          m_data = other.m_data;
        }
        return *this;
      }

      /**
       * @brief Move assignment operator for the iterator.
       * @param other The iterator to move from.
       * @return Reference to the moved iterator.
       */
      iterator &operator=(iterator &&other) noexcept {
        if (this != &other) {
          m_data = std::exchange(other.m_data, nullptr);
        }
        return *this;
      }

      /**
       * @brief Destructor for the iterator.
       */
      virtual ~iterator() noexcept {}

     public:
      /**
       * @brief Pre-increment operator for the iterator.
       * @return Reference to the incremented iterator.
       */
      auto operator++() noexcept -> iterator & {
        ++m_data;
        return *this;
      }

      /**
       * @brief Post-increment operator for the iterator.
       * @return Copy of the iterator before incrementing.
       */
      auto operator++(int) noexcept -> iterator {
        auto tmp = *this;
        ++m_data;
        return tmp;
      }

      /**
       * @brief Pre-decrement operator for the iterator.
       * @return Reference to the decremented iterator.
       */
      auto operator--() noexcept -> iterator & {
        --m_data;
        return *this;
      }

      /**
       * @brief Post-decrement operator for the iterator.
       * @return Copy of the iterator before decrementing.
       */
      auto operator--(int) noexcept -> iterator {
        auto tmp = *this;
        --m_data;
        return tmp;
      }

      /**
       * @brief Compound assignment operator for the iterator.
       * @param i The value to add to the iterator.
       * @return Reference to the updated iterator.
       */
      auto operator+=(size_t i) noexcept -> iterator & {
        m_data += i;
        return *this;
      }

      /**
       * @brief Compound assignment operator for the iterator.
       * @param i The value to subtract from the iterator.
       * @return Reference to the updated iterator.
       */
      auto operator-=(size_t i) noexcept -> iterator & {
        m_data -= i;
        return *this;
      }

      /**
       * @brief Addition operator for the iterator.
       * @param i The value to add to the iterator.
       * @return New iterator with the added value.
       */
      auto operator+(size_t i) const noexcept -> iterator { return iterator(m_data + i); }

      /**
       * @brief Subtraction operator for the iterator.
       * @param i The value to subtract from the iterator.
       * @return New iterator with the subtracted value.
       */
      auto operator-(size_t i) const noexcept -> iterator { return iterator(m_data - i); }

      /**
       * @brief Subtraction operator for two iterators.
       * @param other The other iterator to subtract.
       * @return The difference between the iterators.
       */
      auto operator-(iterator const &other) const noexcept -> size_t { return m_data - other.m_data; }

      /**
       * @brief Dereference operator for the iterator.
       * @return Reference to the value pointed to by the iterator.
       */
      auto operator*() const noexcept -> deref_type { return *m_data; }

      /**
       * @brief Subscript operator for the iterator.
       * @param i The index to access.
       * @return Reference to the value at the specified index.
       */
      auto operator[](size_t i) const noexcept -> deref_type { return m_data[i]; }

      /**
       * @brief Equality comparison operator for two iterators.
       * @tparam OtherConst Flag indicating whether the other iterator is const
       * or not.
       * @param other The other iterator to compare.
       * @return True if the iterators are equal, false otherwise.
       */
      template <bool OtherConst = Const>
      auto operator==(iterator<OtherConst> const &other) const noexcept -> bool {
        return m_data == other.m_data;
      }

      /**
       * @brief Inequality comparison operator for two iterators.
       * @tparam OtherConst Flag indicating whether the other iterator is const
       * or not.
       * @param other The other iterator to compare.
       * @return True if the iterators are not equal, false otherwise.
       */
      template <bool OtherConst = Const>
      auto operator!=(iterator<OtherConst> const &other) const noexcept -> bool {
        return m_data != other.m_data;
      }

      /**
       * @brief Less than or equal to comparison operator for two iterators.
       * @tparam OtherConst Flag indicating whether the other iterator is const
       * or not.
       * @param other The other iterator to compare.
       * @return True if this iterator is less than or equal to the other
       * iterator, false otherwise.
       */
      template <bool OtherConst = Const>
      auto operator<=(iterator<OtherConst> const &other) const noexcept -> bool {
        return m_data <= other.m_data;
      }

      /**
       * @brief Greater than or equal to comparison operator for two iterators.
       * @tparam OtherConst Flag indicating whether the other iterator is const
       * or not.
       * @param other The other iterator to compare.
       * @return True if this iterator is greater than or equal to the other
       * iterator, false otherwise.
       */
      template <bool OtherConst = Const>
      auto operator>=(iterator<OtherConst> const &other) const noexcept -> bool {
        return m_data >= other.m_data;
      }

      /**
       * @brief Less than comparison operator for two iterators.
       * @tparam OtherConst Flag indicating whether the other iterator is const
       * or not.
       * @param other The other iterator to compare.
       * @return True if this iterator is less than the other iterator, false
       * otherwise.
       */
      template <bool OtherConst = Const>
      auto operator<(iterator<OtherConst> const &other) const noexcept -> bool {
        return m_data < other.m_data;
      }

      /**
       * @brief Greater than comparison operator for two iterators.
       * @tparam OtherConst Flag indicating whether the other iterator is const
       * or not.
       * @param other The other iterator to compare.
       * @return True if this iterator is greater than the other iterator, false
       * otherwise.
       */
      template <bool OtherConst = Const>
      auto operator>(iterator<OtherConst> const &other) const noexcept -> bool {
        return m_data > other.m_data;
      }

     public:
      template <typename PtrT>
      operator PtrT *() const {
        return reinterpret_cast<PtrT *>(m_data);
      }
    };

    /**
     * @brief Returns an iterator to the beginning of the column.
     * @return An iterator to the beginning of the column.
     */
    auto begin() const noexcept -> iterator<false> { return iterator<false>(m_data); }

    /**
     * @brief Returns a const iterator to the beginning of the column.
     * @return A const iterator to the beginning of the column.
     */
    auto cbegin() const noexcept -> iterator<true> { return iterator<true>(m_data); }

    /**
     * @brief Returns an iterator to the specified index of the column.
     * @param idx The index to start the iterator from.
     * @return An iterator to the specified index of the column.
     * @throws std::out_of_range Throws if index is out of range
     */
    auto begin(size_t idx) const -> iterator<false> {
      if (valid_index(idx)) [[likely]] {
        return iterator<false>(m_data + idx);
      } else [[unlikely]] {
        throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
      }
    }

    /**
     * @brief Returns a const iterator to the specified index of the column.
     * @param idx The index to start the iterator from.
     * @return A const iterator to the specified index of the column.
     * @throws std::out_of_range Throws if index is out of range
     */
    auto cbegin(size_t idx) const -> iterator<true> {
      if (valid_index(idx)) [[likely]] {
        return iterator<true>(m_data + idx);
      } else [[unlikely]] {
        throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
      }
    }

    /**
     * @brief Returns an iterator to the end of the column.
     * @return An iterator to the end of the column.
     */
    auto end() const noexcept -> iterator<false> { return iterator<false>(m_end); }

    /**
     * @brief Returns a const iterator to the end of the column.
     * @return A const iterator to the end of the column.
     */
    auto cend() const noexcept -> iterator<true> { return iterator<true>(m_end); }

    /**
     * @brief Returns an iterator to the specified index after the end of the
     * column.
     * @param idx The index to end the iterator at.
     * @return An iterator to the specified index after the end of the column.
     * @throws std::out_of_range Throws if index is out of range
     */
    auto end(size_t idx) const -> iterator<false> {
      if (valid_index<true>(idx)) [[likely]] {
        return iterator<false>(m_data + idx);
      } else [[unlikely]] {
        throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
      }
    }

    /**
     * @brief Returns a const iterator to the specified index after the end of
     * the column.
     * @param idx The index to end the iterator at.
     * @return A const iterator to the specified index after the end of the
     * column.
     * @throws std::out_of_range Throws if index is out of range
     */
    auto cend(size_t idx) const -> iterator<true> {
      if (valid_index<true>(idx)) [[likely]] {
        return iterator<true>(m_data + idx);
      } else [[unlikely]] {
        throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
      }
    }
  };
}  // namespace tuddbs

#endif