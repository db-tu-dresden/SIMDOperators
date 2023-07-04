#ifndef SRC_OPERATORS_NATURALEQUIJOIN_HPP
#define SRC_OPERATORS_NATURALEQUIJOIN_HPP

#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>

#include <SIMDOperators/datastructures/hash_based/strategies/linear_probing.h>
#include <SIMDOperators/datastructures/hash_based/hash_utils.h>
#include <SIMDOperators/datastructures/hash_based/hash_map.h>
#include <SIMDOperators/datastructures/hash_based/hash_set.h>
#include <SIMDOperators/MorphStore_old/vector/complex/hash.h>

namespace tuddbs{
    
    template<typename ProcessingStyle, class DataStructure>
    class naturalEquiJoin {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        template<
            typename batchps,
            class DataStructureBatch
        >
        struct equi_join_build_batch {
            // to make it accessable by select class
            friend class naturalEquiJoin<ProcessingStyle, DataStructureBatch>;

            using reg_t = typename batchps::register_type;
            using mask_t = typename batchps::mask_type;
            using imask_t = typename batchps::imask_type;

            //IMPORT_VECTOR_BOILER_PLATE(batchps)
            MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
                base_type *& p_InBuildDataPtr,
                size_t const p_Count,
                base_type const p_InPositionIn,
                DataStructureBatch & hs
            ) {
                using namespace vectorlib;
                reg_t positionVector = tsl::custom_sequence<batchps>(p_InPositionIn, 1);
                reg_t const incrementVector = tsl::set1<batchps>(batchps::vector_element_count());
                auto state = hs.template get_lookup_insert_strategy_state< batchps >();
                for(size_t i = 0; i < p_Count; ++i) {
                    hs.template insert<batchps>(
                    tsl::load<batchps>( p_InBuildDataPtr ),
                    positionVector,
                    state );
                    p_InBuildDataPtr += batchps::vector_element_count();
                    positionVector = tsl::add< batchps >( positionVector, incrementVector );
                }
            }
        };

        template<
            typename batchps,
            class DataStructureBatch
        >
        struct equi_join_probe_batch {
            friend class naturalEquiJoin<ProcessingStyle, DataStructureBatch>;

            using reg_t = typename batchps::register_type;
            using mask_t = typename batchps::mask_type;
            using imask_t = typename batchps::imask_type;

            //IMPORT_VECTOR_BOILER_PLATE(batchps)
            MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t
            apply(
                base_type *& p_InProbeDataPtr,
                size_t const p_Count,
                base_type *& p_OutPosLCol,
                base_type *& p_OutPosRCol,
                base_type const p_InPositionIn,
                DataStructureBatch & hs
            ) {
                using namespace vectorlib;
                auto state = hs.template get_lookup_insert_strategy_state< batchps >();
                size_t resultCount = 0;

                reg_t positionVector = tsl::custom_sequence<batchps>(p_InPositionIn, 1);
                reg_t const incrementVector = tsl::set1<batchps>( batchps::vector_element_count() );
                reg_t lookupResultValuesVector;
                mask_t lookupResultMask;
                imask_t imask;
                uint8_t hitResultCount;

                for( size_t i = 0; i < p_Count; ++i ) {
                    std::tie( lookupResultValuesVector, lookupResultMask, hitResultCount ) =
                    hs.template lookup<batchps>(
                        tsl::load<batchps>( p_InProbeDataPtr ),
                        state
                    );
                    imask = tsl::to_integral<batchps>(lookupResultMask);
                    tsl::compress_store<batchps>(
                    imask, p_OutPosLCol, lookupResultValuesVector);
                    tsl::compress_store<batchps>(
                    imask, p_OutPosRCol, positionVector);
                    p_OutPosRCol += hitResultCount;
                    p_OutPosLCol += hitResultCount;
                    resultCount += hitResultCount;
                    positionVector = tsl::add< batchps >( positionVector, incrementVector );
                    p_InProbeDataPtr += batchps::vector_element_count();
                    hitResultCount = 0;
                    lookupResultMask = 0;
                    imask = 0;
                }

                return resultCount;
            }
        };

    public:

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static std::tuple<col_ptr, col_ptr> apply(const_col_ptr lhs, const_col_ptr rhs){
            
            /// Get the alignment of the columns
            typename AlignmentHelper<ps>::Alignment alignment_lhs = AlignmentHelper<ps>::getAlignment(lhs->getRawDataPtr());
            size_t alignment_elements_lhs;
            if (lhs->getPopulationCount() < alignment_lhs.getElementsUntilAlignment()) {
                alignment_elements_lhs = lhs->getPopulationCount();
            } else {
                alignment_elements_lhs = alignment_lhs.getElementsUntilAlignment();
            }

            typename AlignmentHelper<ps>::Alignment alignment_rhs = AlignmentHelper<ps>::getAlignment(rhs->getRawDataPtr());
            size_t alignment_elements_rhs;
            if (rhs->getPopulationCount() < alignment_rhs.getElementsUntilAlignment()) {
                alignment_elements_rhs = rhs->getPopulationCount();
            } else {
                alignment_elements_rhs = alignment_rhs.getElementsUntilAlignment();
            }

            const size_t inBuildDataCount = lhs->getPopulationCount();
            const size_t inProbeDataCount = rhs->getPopulationCount();

            const size_t outCountEstimate = 0;

            const size_t outCount = bool(outCountEstimate)
                                // use given estimate
                                ? (outCountEstimate)
                                // use pessimistic estimate
                                : (inProbeDataCount);

            base_type * lhs_ptr = const_cast<base_type *>(lhs->getRawDataPtr());
            base_type * rhs_ptr = const_cast<base_type *>(rhs->getRawDataPtr());

            DataStructure hs( inBuildDataCount );

            auto result_lhs = Column<base_type>::create(outCount, ps::vector_size_B());
            auto result_rhs = Column<base_type>::create(outCount, ps::vector_size_B());

            auto result_lhs_ptr = result_lhs->getRawDataPtr();
            auto result_rhs_ptr = result_rhs->getRawDataPtr();

            /// Build phase
            equi_join_build_batch<scalar, DataStructure>::apply( lhs_ptr, alignment_elements_lhs, 0, hs );
            size_t vector_count_build = (lhs->getPopulationCount() - alignment_elements_lhs) / ps::vector_element_count();
            equi_join_build_batch<ProcessingStyle, DataStructure>::apply( lhs_ptr, vector_count_build, alignment_elements_lhs, hs );
            
            equi_join_build_batch<scalar, DataStructure>::apply( lhs_ptr, lhs->getPopulationCount() - alignment_elements_lhs - vector_count_build * ps::vector_element_count(), alignment_elements_lhs + vector_count_build * ps::vector_element_count(), hs );
            
            /// Probe phase
            size_t result_count = equi_join_probe_batch<scalar, DataStructure>::apply( rhs_ptr, alignment_elements_rhs, result_lhs_ptr, result_rhs_ptr, 0, hs );
            
            size_t vector_count_probe = (rhs->getPopulationCount() - alignment_elements_rhs) / ps::vector_element_count();
            result_count += equi_join_probe_batch<ProcessingStyle, DataStructure>::apply( rhs_ptr , vector_count_probe, result_lhs_ptr, result_rhs_ptr, alignment_elements_rhs, hs );
            
            result_count += equi_join_probe_batch<scalar, DataStructure>::apply( rhs_ptr, rhs->getPopulationCount() - alignment_elements_rhs - vector_count_probe * ps::vector_element_count(), result_lhs_ptr, result_rhs_ptr  , alignment_elements_rhs + vector_count_probe * ps::vector_element_count(), hs );
            
            result_lhs->setPopulationCount(result_count);
            result_rhs->setPopulationCount(result_count);

            return std::make_tuple(result_lhs, result_rhs);
        }


    };

    //Define convenience function
    template<
        typename VectorExtension
    >
    std::tuple<
        Column< typename VectorExtension::base_type > *,
        Column< typename VectorExtension::base_type > *
    > 
    natural_equi_join(
    Column< typename VectorExtension::base_type > const * const p_InDataLCol,
    Column< typename VectorExtension::base_type > const * const p_InDataRCol,
    size_t const outCountEstimate = 0
    ) {
    return naturalEquiJoin<
        VectorExtension,
        vectorlib::hash_map<
            VectorExtension,
            vectorlib::multiply_mod_hash,
            vectorlib::size_policy_hash::EXPONENTIAL,
            vectorlib::scalar_key_vectorized_linear_search,
            60
        >
    >::apply(p_InDataLCol,p_InDataRCol);
    }

}; //namespace tuddbs



#endif //SRC_OPERATORS_NATURALEQUIJOIN_HPP
