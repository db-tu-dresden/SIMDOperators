//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_COMPLEX_HASH_H
#define MORPHSTORE_VECTOR_COMPLEX_HASH_H

#include <SIMDOperators/utils/preprocessor.h>

#include <tslintrin.hpp>

namespace vectorlib {



   template< class VectorExtension >
   struct multiply_mod_hash {
      struct state_t {
         typename VectorExtension::register_type const m_Prime;
         state_t( typename VectorExtension::base_type const p_Prime = ( ( 1 << 16 ) + 1 ) ):
            m_Prime{tsl::set1<VectorExtension>( p_Prime ) }{ }
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename VectorExtension::register_type apply(
         typename VectorExtension::register_type const & p_Key,
         state_t const & p_State
      ) {
         return
            tsl::mul<VectorExtension>(
               p_Key,
               p_State.m_Prime
            );
      }
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename VectorExtension::register_type apply(
         typename VectorExtension::register_type const & p_Key,
         typename VectorExtension::register_type const & p_Key2,
         state_t const & p_State
      ) {
         return
            tsl::mul<VectorExtension>(
               tsl::mul<VectorExtension>(
                  p_Key, p_Key2
               ),
               p_State.m_Prime
            );
      }
   };

}
#endif //MORPHSTORE_VECTOR_COMPLEX_HASH_H
