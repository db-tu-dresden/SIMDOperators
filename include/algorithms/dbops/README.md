# SIMD-Database Operators

This folder contains all relevant header files for a variety of operators relevant to database processing.

Operators can be hinted.

## Available Hints

There are several hints that can be used to specify the behavior (**B**) of an operator, its in and output (**I/O**) or enable specific optimizations (**Opt**). The hints must be wrapped into an `OperatorHintSet` as template arguments.
Example:

```cpp
auto GroupBy = Group<tsl::simd<uint64_t, avx2>, 
                     OperatorHintset<hints::hashing::size_exp_2,
                                     hints::hashing::keys_may_contain_zero,
                                     hints::hashing::linear_displacement,
                                     hints::grouping::global_first_occurence_required>>;
GroupBy::builder_t builder(array_keys, array_gids, array_first_positions, 128);
/**...*/
```

|Hint|Type|Effect|Defined in|
|--|--|--|--|
|`hints::operators::preserve_original_positions`|**B**| |simdops.hpp|
|`hints::intermediate::position_list`|**I/O**|  |simdops.hpp|
|`hints::intermediate::bit_mask`|**I/O**|  |simdops.hpp|
|`hints::intermediate::dense_bit_mask`|**I/O**|  |simdops.hpp|
|`hints::hashing::unique_keys`|**Opt**|  |hashing.hpp|
|`hints::hashing::size_exp_2`|**Opt**|  |hashing.hpp|
|`hints::hashing::keys_may_contain_zero`|**B**|  |hashing.hpp|
|`hints::hashing::is_hull_for_merging`|**Opt**|  |hashing.hpp|
|`hints::hashing::linear_displacement`|**B**|  |hashing.hpp|
|`hints::hashing::refill`|**B**|  |hashing.hpp|
|`hints::grouping::global_first_occurence_required`|**B**|  |group.hpp|
