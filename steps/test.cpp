#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>

#include "NumCpp.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"

int main() {
  nc::NdArray<double> data = {0.0};
  std::cout << data.data() << std::endl;
}