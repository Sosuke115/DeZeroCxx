#include <iostream>

#include "NumCpp.hpp"

int main() {
  nc::NdArray<double> data({0.5});
  nc::NdArray<double> data2;

  data2 = data;

  data[0] = 1.0;
  std::cout << data << std::endl;   // 1.0
  std::cout << data2 << std::endl;  // 0.5

  // copyが走ってる
}