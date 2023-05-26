#ifndef SRC_OPERATORS_SELECT_HPP
#define SRC_OPERATORS_SELECT_HPP

#include "utils/preprocessor.h"
#include "utils/AlignmentHelper.hpp"

#include <iostream>

#include <datastructure/column.hpp>

using namespace std;

namespace tuddbs{
    template<typename ProcessingStyle, template < typename > typename CompareOperator >
    class select {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = std::shared_ptr<col_t>;
        using const_col_ptr = std::shared_ptr<const col_t>;

        template<typename batchps>
        class batch {
            // MSV_CXX_ATTRIBUTE_FORCE_INLINE
            // static void apply(col_t *& result, col_t *& column, base_type predicate, ){

            // }
        };

        template<typename kernelps>
        class kernel {
            uint8_t dummy;

            // MSV_CXX_ATTRIBUTE_FORCE_INLINE
            // static 5
        };

    public:

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr apply(const_col_ptr column, base_type predicate){
            
            /// Get the alignment of the column
            auto alignment = AlignmentHelper<ps>::getAlignment(column->getRawDataPtr());


            auto result = Column<base_type>::create(column->getLength(), ps::vector_size_B());

            /// Scalar preprocessing



        }


    };

}; //namespace tuddbs



#endif //SRC_OPERATORS_SELECT_HPP
