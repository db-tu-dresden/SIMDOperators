//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H

//#include <SIMDOperators/MorphStore_old/core/memory/mm_glob.h>
#include <SIMDOperators/utils/preprocessor.h>
#include <tslintrin.hpp>
#include <SIMDOperators/datastructures/hash_based/hash_utils.h>

#ifdef MSV_NO_SELFMANAGED_MEMORY
#include <SIMDOperators/MorphStore_old/core/memory/management/utils/alignment_helper.h>
#endif

#include <tuple> //tuple
#include <cstdint> //uint8_t
#include <cstddef> //size_t
#include <algorithm> //std::fill

namespace vectorlib {

   /**
    * Hash set constant size (NO RESIZING), linear probing
    * @tparam VectorExtension
    * @tparam HashFunction
    * @tparam MaxLoadfactor
    */
   template<
      class BiggestSupportedVectorExtension,
      template<class> class HashFunction,
      size_policy_hash SPH,
      template<class, class, template<class>class, size_policy_hash> class LookupInsertStrategy,
      size_t MaxLoadfactor //60 if 0.6...
   >
   class hash_map{
   public:

      template< class VectorExtension >
      DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
      void
      insert(
         typename VectorExtension::register_type const & p_KeysToLookup,
         typename VectorExtension::register_type const & p_Values,
         typename
         LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
         & p_LookupInsertStrategyState
      ) {
         LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::insert(
            p_KeysToLookup,
            p_Values,
            p_LookupInsertStrategyState
         );
      }

      template< class VectorExtension >
      DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
      std::tuple<
         typename VectorExtension::register_type,      // groupID vector register
         typename VectorExtension::register_type,      // groupExt vector register
         typename VectorExtension::mask_type, // active groupExt elements
         uint8_t        // Number of active groupExt elements
      >
      insert_and_lookup(
         typename VectorExtension::register_type const & p_InKeyVector,
         typename VectorExtension::base_type & p_InStartPosFromKey,
         typename VectorExtension::base_type & p_InStartValue,
         typename LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
         & p_LookupInsertStrategyState
      ) {
         return LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::insert_and_lookup(
            p_InKeyVector,
            p_InStartPosFromKey,
            p_InStartValue,
            p_LookupInsertStrategyState
         );
      }


      template< class VectorExtension >
      DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
      std::tuple< typename VectorExtension::register_type, typename VectorExtension::mask_type, uint8_t >
      lookup(
         typename VectorExtension::register_type const & p_KeysToLookup,
         typename
         LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
         & p_LookupInsertStrategyState
      ) {
         return LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::lookup(
            p_KeysToLookup,
            p_LookupInsertStrategyState
         );
      }


   private:
      size_helper<BiggestSupportedVectorExtension, MaxLoadfactor, SPH> const m_SizeHelper;

      typename BiggestSupportedVectorExtension::base_type * const m_Keys;
      typename BiggestSupportedVectorExtension::base_type * const m_Values;
   public:
      hash_map(
         size_t const p_DistinctElementCountEstimate
      ) :
         m_SizeHelper{
            p_DistinctElementCountEstimate
         },
         m_Keys{
            new (std::align_val_t(BiggestSupportedVectorExtension::vector_size_B())) typename BiggestSupportedVectorExtension::base_type[m_SizeHelper.m_Count]
         },
         m_Values{
            new (std::align_val_t(BiggestSupportedVectorExtension::vector_size_B())) typename BiggestSupportedVectorExtension::base_type[m_SizeHelper.m_Count]
         }
      {
         std::fill(m_Keys, m_Keys+m_SizeHelper.m_Count, 0);
         std::fill(m_Values, m_Values+m_SizeHelper.m_Count, 0);
      }


      typename BiggestSupportedVectorExtension::base_type * get_data_keys( void ) {
         return m_Keys;
      }

      typename BiggestSupportedVectorExtension::base_type * get_data_values( void ) {
         return m_Values;
      }

      size_t get_bucket_count( void ) {
         return m_SizeHelper.m_Count;
      }

      template< class VectorExtension >
      using strategy_state =
         typename
         LookupInsertStrategy<
            VectorExtension,
            BiggestSupportedVectorExtension,
            HashFunction,
            SPH
         >::state_single_key_single_value_t;

      template< class VectorExtension >
      strategy_state<VectorExtension>
      get_lookup_insert_strategy_state( void ) {
         return
            typename
            LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t(
               m_Keys,
               m_Values,
               m_SizeHelper.m_Count
            );
      }



      ~hash_map() = default;

      void print( void ) const {
         uint64_t mulres, resizeres, alignres;
         fprintf( stdout, "HashSet idx;Key;Key*Prime;Resized;Aligned (StartPos)\n");
         for( size_t i = 0; i < m_SizeHelper.m_Count; ++i ) {
            __builtin_umull_overflow( m_Keys[i], 65537, &mulres);
            resizeres = mulres & 1023;
            alignres = resizeres & (typename BiggestSupportedVectorExtension::base_type)~(BiggestSupportedVectorExtension::vector_element_count() - 1);
            fprintf( stdout, "%lu;%lu;%lu;%lu,%lu\n", i, m_Keys[i],mulres,resizeres,alignres );
         }
      }
   };

}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H
