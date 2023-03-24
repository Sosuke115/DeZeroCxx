#include <iostream>

#include "NumCpp.hpp"

class Variable {
 public:
  nc::NdArray<double> data_;
  Variable(const nc::NdArray<double>& data);
};

Variable::Variable(const nc::NdArray<double>& data) : data_(data) {}

int main() {
  nc::NdArray<double> data = {2.0};
  auto x = Variable(data);
  std::cout << x.data_ << std::endl;
}