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
 * @file iterable.hpp
 * @brief This file contains the concept definitions for iterable classes.
 */

#ifndef SIMDOPS_INCLUDE_ITERABLE_HPP
#define SIMDOPS_INCLUDE_ITERABLE_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "tsl.hpp"

namespace tuddbs {

  namespace hints {
    namespace memory {
      struct aligned {};
    }  // namespace memory
  }    // namespace hints

  /**
   * @concept Integral
   * @brief Concept that checks if a type is integral.
   * @tparam T The type to check.
   */
  template <typename T>
  concept Integral = std::is_integral_v<T>;

  /**
   * @concept Unsigned
   * @brief Concept that checks if a type is unsigned integral.
   * @tparam T The type to check.
   */
  template <typename T>
  concept Unsigned = Integral<T> && std::is_unsigned_v<T>;

  /**
   * @concept PointerType
   * @brief Concept that checks if a type is a pointer.
   * @tparam T The type to check.
   */
  template <typename T>
  concept PointerType = std::is_pointer_v<T>;

  /**
   * @concept ClassOrArithmeticPointer
   * @brief Concept that checks if a type is either the same as another type or
   * a TSLArithmeticPointer.
   * @tparam T The type to check.
   * @tparam U The other type to compare with.
   */
  template <typename T, typename U>
  concept ClassOrArithmeticPointer = std::is_same_v<U, T> || tsl::TSLArithmeticPointer<T>;

  /**
   * @concept SimdOpsIterableClass
   * @brief Concept that checks if a class satisfies the requirements of a
   * SimdOps iterable class.
   * @tparam T The class type to check.
   */
  template <typename T>
  concept SimdOpsIterableClass = requires(T &t) {
    { t.operator*() } -> tsl::TSLArithmeticReference;
  } && requires(T &t, size_t i) {
    { t.operator[](i) } -> tsl::TSLArithmeticReference;
    { t.operator+(i) } -> ClassOrArithmeticPointer<T>;
    { t.operator-(i) } -> ClassOrArithmeticPointer<T>;
  } && requires(T const &t) {
    { t.operator*() } -> tsl::TSLArithmeticReference;
  } && requires(T const &t1, T const &t2) {
    { t1.operator!=(t2) } -> std::same_as<bool>;
    { t1.operator==(t2) } -> std::same_as<bool>;
    { t1.operator<=(t2) } -> std::same_as<bool>;
    { t1.operator>=(t2) } -> std::same_as<bool>;
    { t1.operator<(t2) } -> std::same_as<bool>;
    { t1.operator>(t2) } -> std::same_as<bool>;
  };

  /**
   * @concept SimdOpsIterable
   * @brief Concept that checks if a type satisfies the requirements of a
   * SimdOps iterable.
   * @tparam T The type to check.
   */
  template <typename T>
  concept SimdOpsIterable = (tsl::TSLArithmeticPointer<T> || SimdOpsIterableClass<T>);

  /**
   * @concept SimdOpsIterableOrSizeT
   * @brief Concept that checks if a type satisfies the requirements of a
   * SimdOps iterable class or is an unsigned integral type.
   * @tparam T The type to check.
   */
  template <typename T>
  concept SimdOpsIterableOrSizeT = SimdOpsIterable<T> || std::is_unsigned_v<T>;

  /**
   * Calculates the end iterator for the given range.
   *
   * @param data The start iterator of the range.
   * @param end The end iterator or size of the range.
   * @return The calculated end iterator.
   * @throws std::invalid_argument if the start iterator is after the end
   * iterator.
   */
  constexpr static auto iter_end(SimdOpsIterable auto data, SimdOpsIterableOrSizeT auto end) {
    if constexpr (std::is_unsigned_v<decltype(end)>) {
      return data + end;
    } else {
      if (data > end) {
        throw std::invalid_argument("Begin is after end");
      }
      return end;
    }
  }

  template <tsl::VectorProcessingStyle SimdStyle>
  constexpr static auto simd_iter_end(SimdOpsIterable auto data, SimdOpsIterableOrSizeT auto end) {
    if constexpr (std::is_unsigned_v<decltype(end)>) {
      return data + (end - (end & (SimdStyle::vector_element_count() - 1)));
    } else {
      if (data > end) {
        throw std::invalid_argument("Begin is after end");
      }
      const size_t dist = end - data;
      return data + (dist - (dist & (SimdStyle::vector_element_count() - 1)));
      // return end;
    }
  }

  template <unsigned long N>
  constexpr static auto batched_iter_end(SimdOpsIterable auto data, SimdOpsIterableOrSizeT auto end) {
    if constexpr (std::is_unsigned_v<decltype(end)>) {
      if constexpr ((N & (N - 1)) == 0) {
        return data + (end - (end & (N - 1)));
      } else {
        return data + (end - (end % N));
      }
    } else {
      if (data > end) {
        throw std::invalid_argument("Begin is after end");
      }
      const size_t dist = end - data;
      if constexpr ((N & (N - 1)) == 0) {
        return data + (dist - (dist & (N - 1)));
      } else {
        return data + (dist - (dist % N));
      }
    }
  }

  constexpr static auto batched_iter_end(SimdOpsIterable auto data, SimdOpsIterableOrSizeT auto end, unsigned long N) {
    if ((N & (N - 1)) == 0) {
      return data + (end - (end & (N - 1)));
    } else {
      return data + (end - (end % N));
    }
  }

  template <tsl::TSLArithmeticPointer To, SimdOpsIterable From>
  constexpr auto reinterpret_iterable(From data) {
    if constexpr (std::is_pointer_v<std::decay_t<From>>) {
      return reinterpret_cast<To>(data);
    } else if constexpr (std::is_convertible_v<From, To>) {
      return static_cast<To>(data);
    } else {
      throw std::invalid_argument("Cannot convert");
    }
  }

}  // namespace tuddbs

#endif  // SIMDOPS_INCLUDE_ITERABLE_HPP