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
#ifndef SRC_SIMDOPERATORS_SIMDOPERATORS_H
#define SRC_SIMDOPERATORS_SIMDOPERATORS_H
#include <SIMDOperators/utils/preprocessor.h>

namespace tuddbs{};

#include <SIMDOperators/datastructures/column.hpp>

#include <SIMDOperators/operators/select.hpp>
#include <SIMDOperators/operators/project.hpp>
#include <SIMDOperators/operators/calc.hpp>
#include <SIMDOperators/operators/MetaOperator.hpp>
#include <SIMDOperators/operators/aggregate.hpp>
#include <SIMDOperators/operators/naturalEquiJoin.hpp>

#endif //SRC_SIMDOPERATORS_SIMDOPERATORS_H