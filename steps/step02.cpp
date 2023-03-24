#include <iostream>

#include "NumCpp.hpp"

class Variable {
 public:
  nc::NdArray<double> data_;
  Variable(const nc::NdArray<double>& data);
};

class Function {
 public:
  Variable operator()(Variable& input) {
    auto x = input.data_;
    auto y = this->forward(x);
    auto output = Variable(y);
    return output;
  }

  virtual nc::NdArray<double> forward(nc::NdArray<double> x) = 0;
};

class Square : public Function {
 public:
  nc::NdArray<double> forward(nc::NdArray<double> x) override { return x * x; }
};

Variable::Variable(const nc::NdArray<double>& data) : data_(data) {}

int main() {
  nc::NdArray<double> data({10.0});
  Square f;
  auto x = Variable(data);
  auto y = f(x);

  std::cout << y.data_ << std::endl;
}