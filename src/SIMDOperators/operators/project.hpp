#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>

template<typename ProcessingStyle>
class project {
    using ps = ProcessingStyle;

    class kernel {


        // MSV_CXX_ATTRIBUTE_FORCE_INLINE
        // static 
    };

  public:

    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static void apply(){
        std::cout << "Project test" << std::endl;
    }


};



