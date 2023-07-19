// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Copyright (c) 2022 SimdOperators Team.
   
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

#ifndef SRC_SIMDOPERATORS_UTILS_CONSTEXPR_MEMBERDETECTOR_H
#define SRC_SIMDOPERATORS_UTILS_CONSTEXPR_MEMBERDETECTOR_H

#include <type_traits>
#include <experimental/type_traits>

namespace tuddbs {
    /**
     * @brief Member detector following the [detector idiom](https://benjaminbrock.net/blog/detection_idiom.php)
     */
    namespace detector {

        template<typename T>
        using detect_member_method_apply = decltype(std::declval<T>().apply());

        template<typename T>
        using detect_static_method_apply = decltype(T::apply);

        /**
         * @brief Detects if a class has a member method called apply
         * 
         * @tparam T Class to check
         */
        template<typename T>
        inline constexpr
        bool has_member_method_apply_v = std::experimental::is_detected_v<detect_member_method_apply, T>;

        /**
         * @brief Detects if a class has a static method called apply
         * 
         * @tparam T Class to check
         */
        template<typename T>
        inline constexpr
        bool has_static_method_apply_v = std::experimental::is_detected_v<detect_static_method_apply, T>;

        /**
         * @brief Detects if a class has a non-static member method called apply
         * 
         * @tparam T Class to check
         */
        template<typename T>
        inline constexpr
        bool has_non_static_method_apply_v = has_member_method_apply_v<T> && !has_static_method_apply_v<T>;


        namespace detail {
            template <class Default, class AlwaysVoid, template<class...> class Op, class... Args>
            struct detector {
                using value_t = std::false_type;
                using type = Default;
            };
            
            template <class Default, template<class...> class Op, class... Args>
            struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
                using value_t = std::true_type;
                using type = Op<Args...>;
            };
            
            struct nonesuch{};

            template <template<class...> class Op, class... Args>
            using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;
        } // namespace detail

    }
}

#endif //SRC_SIMDOPERATORS_UTILS_CONSTEXPR_MEMBERDETECTOR_H
