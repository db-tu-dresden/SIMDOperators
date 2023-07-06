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

/**
 * This is a small example how to use the SimdOperators library.
 * 
 */


#include <iostream>

// Include the SIMD operators main header file.
// This includes all operators present in the library.
#include <SIMDOperators/SIMDOperators.h>



int main(){
    using namespace std;
    using namespace tuddbs;

    // Define the processing style to use.
    #ifdef USE_AVX512
        using ps = typename tsl::simd<uint64_t, tsl::avx512>;
    #else
        using ps = typename tsl::simd<uint64_t, tsl::scalar>;
    #endif


    // Create a column with 100 elements of type uint64_t.
    // The column is aligned to the vector size of the processing style.
    auto col = new Column<uint64_t>(100, ps::vector_size_B());
    // Set the population count to 100.
    col->setPopulationCount(100);
    // Fill the column with values.
    {
        auto data = col->getData();
        for (int i = 0; i < col->getLength(); ++i) {
            data[i] = i;
        }
    }

    // Select all values greater than 50.
    // Apply the select operator.
    auto select_res = tuddbs::select<ps, tsl::functors::greater_than>::apply(col, 50);

    // Print the result.
    cout << "Result of select operator:" << endl;
    cout << "Population count: " << select_res->getPopulationCount() << endl;
    cout << "Data: " << endl;
    auto data = select_res->getData();
    for (int i = 0; i < select_res->getPopulationCount(); ++i) {
        cout << data[i] << endl;
    }


}