#ifndef SRC_OPERATORS_PROJECTIONPATH_HPP
#define SRC_OPERATORS_PROJECTIONPATH_HPP

#include <iostream>

#include <SIMDOperators/operators/project.hpp>

namespace tuddbs{
    template<typename ProcessingStyle>
    class projectionPath {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

    
    public:

        template<typename... Args> 
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr apply(const_col_ptr column, Args... positions){
            const_col_ptr current_pos = nullptr;
            for (const_col_ptr pos : {positions...}) {
                if (!current_pos) {
                    current_pos = pos;
                    continue;
                }
                current_pos = tuddbs::project<ps>::apply(pos, current_pos);
                
            }

            return tuddbs::project<ps>::apply(column, current_pos);
        }


    };

}; //namespace tuddbs

#endif //SRC_OPERATORS_PROJECTIONPATH_HPP