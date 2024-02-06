#include "static/utils/type_concepts.hpp"
#include "tslintrin.hpp"

#include <vector>
#include <utility> // std::exchange

namespace tuddbs{
  template<typename T> concept ReferenceOrArithmeticPointer = std::is_reference_v<T> || tsl::TSLArithmeticPointer<T>;
  template<typename T>
  concept SimdOpsIterableClass = requires(T & t) {
    { t.operator*() } -> tsl::TSLArithmeticReference;
  } && requires(T & t, size_t i) {
    { t.operator[](i) } -> tsl::TSLArithmeticReference;
    { t.operator+(i) }  -> ReferenceOrArithmeticPointer;
    { t.operator-(i) }  -> ReferenceOrArithmeticPointer;
  } && requires(T const & t) {
    { t.operator*() } -> tsl::TSLConstArithmetic;
  } && requires(T const & t1, T const & t2) {
    { t1.operator!=(t2) } -> std::same_as<bool>;
    { t1.operator==(t2) } -> std::same_as<bool>;
    { t1.operator<=(t2) } -> std::same_as<bool>;
    { t1.operator>=(t2) } -> std::same_as<bool>;
    { t1.operator<(t2) } -> std::same_as<bool>;
    { t1.operator>(t2) } -> std::same_as<bool>;
  };
  template<typename T> concept SimdOpsIterable = tsl::TSLArithmeticPointer<T> || SimdOpsIterableClass<T>;
  template<typename T>
  concept SimdOpsIterableOrSizeT = SimdOpsIterableClass<T> || std::is_unsigned_v<T>;


  template<tsl::TSLArithmetic T>
  class Column {
    private:
      T * const m_data;
      T * const m_end;
      size_t    m_count;
      size_t    m_size;
    public:
      explicit Column(size_t count)
      : m_data(reinterpret_cast<T*>(new T[count])),
        m_end(m_data + m_count),
        m_count(count),
        m_size(count*sizeof(T)) {}
      Column(Column const &) = delete;
      Column(Column && other) noexcept
      : m_data(std::exchange(other.m_data, nullptr)),
        m_end(std::exchange(other.m_end, nullptr)),
        m_count(std::exchange(other.m_count, 0)) {}
      Column & operator=(Column const &) = delete;
      Column & operator=(Column && other) noexcept {
        if (this != &other) {
          delete[] m_data;
          m_data = std::exchange(other.m_data, nullptr);
          m_end = std::exchange(other.m_end, nullptr);
          m_count = std::exchange(other.m_count, 0);
        }
        return *this;
      }
      virtual ~Column() { delete[] m_data; }
    public:
      class View {
        private:
          T * m_data;
        public:
          explicit View(T * data) : m_data(data) {}
          View(View const & other): m_data(other.m_data) {};
          View(View && other) noexcept
          : m_data(std::exchange(other.m_data, nullptr)) {}
          View & operator=(View const & other) {
            if (this != &other) {
              m_data = other.m_data;
            }
            return *this; 
          }
          View & operator=(View && other) noexcept {
            if (this != &other) {
              m_data = std::exchange(other.m_data, nullptr);
            }
            return *this;
          }
          virtual ~View() {}
        public:
          auto operator++() -> View & { ++m_data; return *this; }
          auto operator++(int) -> View { auto tmp = *this; ++m_data; return tmp; }
          auto operator--() -> View & { --m_data; return *this; }
          auto operator--(int) -> View { auto tmp = *this; --m_data; return tmp; }
          auto operator+=(size_t i) -> View & { m_data += i; return *this; }
          auto operator-=(size_t i) -> View & { m_data -= i; return *this; }
          auto operator+(size_t i) const -> View { return View(m_data + i); }
          auto operator-(size_t i) const -> View { return View(m_data - i); }
          auto operator-(View const & other) const -> size_t { return m_data - other.m_data; }
          auto operator*() const -> T & { return *m_data; }
          auto operator[](size_t i) const -> T & { return m_data[i]; }
          auto operator==(View const & other) const -> bool { return m_data == other.m_data; }
          auto operator!=(View const & other) const -> bool { return m_data != other.m_data; }
          auto operator<=(View const & other) const -> bool { return m_data <= other.m_data; }
          auto operator>=(View const & other) const -> bool { return m_data >= other.m_data; }
          auto operator<(View const & other) const -> bool { return m_data < other.m_data; }
          auto operator>(View const & other) const -> bool { return m_data > other.m_data; }
      };
      auto begin() -> View { return View(m_data); }
      auto end() -> View { return View(m_end); }  
      auto begin() const -> View { return View(m_data); }
      auto end() const -> View { return View(m_end); }
      auto size() const -> size_t { return m_size; }
      auto count() const -> size_t { return m_count; }
      auto partition(size_t number_of_partitions) -> std::vector<View> {
        std::vector<View> partitions;
        size_t partition_size = m_count / number_of_partitions;
        for (size_t i = 0; i < number_of_partitions; ++i) {
          partitions.push_back(View(m_data + i * partition_size));
        }
        if (m_count % number_of_partitions != 0) {
          partitions.push_back(View(m_data + number_of_partitions * partition_size));
        }
        return partitions;
      }
  };

template<typename T>
concept data_ptr_like = requires(T & t) {
  { t.operator*() } -> arithmetic_reference;
} && requires(T & t, size_t i) {
  { t.operator[](i) } -> arithmetic_reference;
  { t.operator+(i) } -> arithmetic_pointer;
  { t.operator-(i) } -> arithmetic_pointer;
} && requires(T const & t) {
  { t.operator*() } -> arithmetic;
} && requires(T const & t, size_t i) {
  { t.operator+(i) } -> const_arithmetic_pointer;
  { t.operator-(i) } -> const_arithmetic_pointer;
  /*{ t.operator!=(std::declval<decltype(t.operator+(i))>())} -> std::same_as<bool>;  
  { t.operator==(std::declval<decltype(t.operator+(i))>())} -> std::same_as<bool>;  
  { t.operator<=(std::declval<decltype(t.operator+(i))>())} -> std::same_as<bool>;  
  { t.operator>=(std::declval<decltype(t.operator+(i))>())} -> std::same_as<bool>;  
  { t.operator<(std::declval<decltype(t.operator+(i))>())} -> std::same_as<bool>;  
  { t