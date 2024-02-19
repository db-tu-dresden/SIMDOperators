

#include <bit>

#include "algorithms/dbops/hashing.hpp"
#include "algorithms/dbops/simdops.hpp"
#include "datastructures/column.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  template <tsl::TSLArithmetic KeyT, tsl::TSLArithmetic ValueT, class HintSet = OperatorHintSet<>,
            class KeyAllocator = KeyT *(*)(size_t), class KeyDeleter = void (*)(KeyT *),
            class ValueAllocator = KeyAllocator, class ValueDeleter = KeyDeleter>
  class HashMap_SimpleValue {
   public:
    using key_type = KeyT;
    using key_pointer_type = KeyT *;
    using key_allocator_type = KeyAllocator;
    using key_deleter_type = KeyDeleter;

    using value_type = ValueT;
    using value_pointer_type = ValueT *;
    using value_allocator_type = ValueAllocator;
    using value_deleter_type = ValueDeleter;

   private:
    tuddbs::InMemoryColumn<KeyT, key_allocator_type, key_deleter_type> m_keys_sink;
    tuddbs::InMemoryColumn<ValueT, value_allocator_type, value_deleter_type> m_values_sink;
    KeyT const m_empty_bucket_value;
    ValueT const m_invalid_value;
    size_t const m_bucket_count = 0;
    size_t m_distinct_key_count = 0;

   public:
    explicit HashMap_SimpleValue(size_t const p_estimated_unique_keys, KeyT const p_empty_bucket_value = (KeyT)0,
                                 ValueT const p_invalid_value = (ValueT)0, bool const p_initialize = true) noexcept
      : m_keys_sink(p_estimated_unique_keys),
        m_values_sink(p_estimated_unique_keys),
        m_empty_bucket_value(p_empty_bucket_value),
        m_invalid_value(p_invalid_value),
        m_bucket_count(p_estimated_unique_keys) {
      if (p_initialize) {
        for (auto &key : m_keys_sink) {
          key = p_empty_bucket_value;
        }
        for (auto &value : m_values_sink) {
          value = p_invalid_value;
        }
      }
    }

    explicit HashMap_SimpleValue(size_t p_estimated_unique_keys, key_allocator_type &&key_allocator,
                                 key_deleter_type &&key_deleter, value_allocator_type &&value_allocator,
                                 value_deleter_type &&value_deleter, KeyT const p_empty_bucket_value = (KeyT)0,
                                 ValueT const p_invalid_value = (ValueT)0, bool const p_initialize = true) noexcept
      : m_keys_sink(p_estimated_unique_keys, std::forward<key_allocator_type>(key_allocator),
                    std::forward<key_deleter_type>(key_deleter)),
        m_values_sink(p_estimated_unique_keys, std::forward<value_allocator_type>(value_allocator),
                      std::forward<value_deleter_type>(value_deleter)),
        m_empty_bucket_value(p_empty_bucket_value),
        m_invalid_value(p_invalid_value),
        m_bucket_count(p_estimated_unique_keys) {
      if (p_initialize) {
        for (auto &key : m_keys_sink) {
          key = p_empty_bucket_value;
        }
        for (auto &value : m_values_sink) {
          value = p_invalid_value;
        }
      }
    }

    explicit HashMap_SimpleValue(key_pointer_type &&p_keys, value_pointer_type &&p_values, size_t count,
                                 key_deleter_type &&key_deleter, value_deleter_type &&value_deleter,
                                 KeyT const p_empty_bucket_value = (KeyT)0, ValueT const p_invalid_value = (ValueT)0,
                                 bool const p_initialize = true) noexcept
      : m_keys_sink(std::forward<key_pointer_type>(p_keys), count, std::forward<key_deleter_type>(key_deleter)),
        m_values_sink(std::forward<value_pointer_type>(p_values), count,
                      std::forward<value_deleter_type>(value_deleter)),
        m_empty_bucket_value(p_empty_bucket_value),
        m_invalid_value(p_invalid_value),
        m_bucket_count(count) {
      if (p_initialize) {
        for (auto &key : m_keys_sink) {
          key = p_empty_bucket_value;
        }
        for (auto &value : m_values_sink) {
          value = p_invalid_value;
        }
      }
    }

    explicit HashMap_SimpleValue(key_pointer_type p_keys, value_pointer_type p_values, size_t count,
                                 KeyT const p_empty_bucket_value = (KeyT)0, ValueT const p_invalid_value = (ValueT)0,
                                 bool const p_initialize = true) noexcept
      : m_keys_sink(p_keys, count),
        m_values_sink(p_values, count),
        m_empty_bucket_value(p_empty_bucket_value),
        m_invalid_value(p_invalid_value),
        m_bucket_count(count) {
      if (p_initialize) {
        for (auto &key : m_keys_sink) {
          key = p_empty_bucket_value;
        }
        for (auto &value : m_values_sink) {
          value = p_invalid_value;
        }
      }
    }

    HashMap_SimpleValue(HashMap_SimpleValue const &) = delete;
    HashMap_SimpleValue(HashMap_SimpleValue &&) = default;
    HashMap_SimpleValue &operator=(HashMap_SimpleValue const &) = delete;
    HashMap_SimpleValue &operator=(HashMap_SimpleValue &&) = default;
    ~HashMap_SimpleValue() = default;

   public:  // Iterators
    auto keys_begin() const noexcept { return m_keys_sink.begin(); }
    auto ckeys_begin() const noexcept { return m_keys_sink.cbegin(); }
    auto keys_end() const noexcept { return m_keys_sink.end(); }
    auto ckeys_end() const noexcept { return m_keys_sink.cend(); }
    auto values_begin() const noexcept { return m_values_sink.begin(); }
    auto cvalues_begin() const noexcept { return m_values_sink.cbegin(); }

   public:  // Allocator / Deleter
    auto key_allocator() const noexcept { return m_keys_sink.allocator(); }
    auto key_deleter() const noexcept { return m_keys_sink.deleter(); }
    auto value_allocator() const noexcept { return m_values_sink.allocator(); }
    auto value_deleter() const noexcept { return m_values_sink.deleter(); }

   public:  // Member access
    auto distinct_key_count() const noexcept { return m_distinct_key_count; }
    void increment_key_count() noexcept { ++m_distinct_key_count; }
    void increment_key_count(size_t p_increment) noexcept { m_distinct_key_count += p_increment; }
    void set_distinct_key_count(size_t p_distinct_key_count) noexcept { m_distinct_key_count = p_distinct_key_count; };

    auto empty_bucket() const noexcept { return m_empty_bucket_value; }
    auto invalid_value() const noexcept { return m_invalid_value; }
    auto bucket_count() const noexcept { return m_bucket_count; }

   public:
    void compactify() noexcept {
      auto keys = keys_begin();
      auto values = values_begin();
      auto keys_it = keys_begin();
      auto keys_end = ckeys_end();
      auto values_it = values_begin();
      for (; keys_it != keys_end; ++keys_it, ++values_it) {
        auto key = *keys_it;
        if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
          if (key != m_empty_bucket_value) {
            *keys = key;
            *values = *values_it;
            ++keys;
            ++values;
            *keys_it = m_empty_bucket_value;
            *values_it = m_invalid_value;
          } else {
            auto const value = *values_it;
            if (value != m_invalid_value) {
              *keys = key;
              *values = value;
              ++keys;
              ++values;
              *keys_it = m_empty_bucket_value;
              *values_it = m_invalid_value;
            }
          }
        } else {
          if (key != m_empty_bucket_value) {
            *keys = key;
            *values = *values_it;
            ++keys;
            ++values;
            *keys_it = m_empty_bucket_value;
            *values_it = m_invalid_value;
          }
        }
      }
    }
  };

  template <tsl::TSLArithmetic KeyT, tsl::TSLArithmetic ValueT, class HintSet = OperatorHintSet<>,
            class KeyAllocator = KeyT *(*)(size_t), class KeyDeleter = void (*)(KeyT *),
            class ValueAllocator = KeyAllocator, class ValueDeleter = KeyDeleter>
  auto create_compact_hashmap(HashMap_SimpleValue<KeyT, ValueT, HintSet, KeyAllocator, KeyDeleter, ValueAllocator,
                                                  ValueDeleter> const &p_hashmap) {
    auto const empty_bucket_value = p_hashmap.empty_bucket();
    auto const invalid_value = p_hashmap.invalid_value();
    auto map_element_count = p_hashmap.distinct_key_count();
    if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
      map_element_count = std::bit_ceil(map_element_count);
    }
    HashMap_SimpleValue<KeyT, ValueT, HintSet, KeyAllocator, KeyDeleter, ValueAllocator, ValueDeleter> result(
      map_element_count, p_hashmap.key_allocator(), p_hashmap.key_deleter(), p_hashmap.value_allocator(),
      p_hashmap.value_deleter(), empty_bucket_value, invalid_value, false);

    auto keys = result.keys_begin();
    auto values = result.values_begin();

    auto keys_it = p_hashmap.ckeys_begin();
    auto const keys_end = p_hashmap.ckeys_end();
    auto values_it = p_hashmap.cvalues_begin();
    for (; keys_it != keys_end; ++keys_it, ++values_it) {
      auto const key = *keys_it;
      if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
        if (key != empty_bucket_value) {
          *keys = key;
          *values = *values_it;
          ++keys;
          ++values;
        } else {
          auto const value = *values_it;
          if (value != invalid_value) {
            *keys = key;
            *values = value;
            ++keys;
            ++values;
          }
        }
      } else {
        if (key != empty_bucket_value) {
          *keys = key;
          *values = *values_it;
          ++keys;
          ++values;
        }
      }
    }
    result.set_distinct_key_count(keys - result.keys_begin());
    return result;
  }

}  // namespace tuddbs