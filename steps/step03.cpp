#include <cmath>
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

class Exp : public Function {
 public:
  nc::NdArray<double> forward(nc::NdArray<double> x) override { return exp(x); }
};

Variable::Variable(const nc::NdArray<double>& data) : data_(data) {}

int main() {
  nc::NdArray<double> data({0.5});
  Square A;
  Exp B;
  Square C;
  auto x = Variable(data);
  auto a = A(x);
  auto b = B(a);
  auto y = C(b);

  std::cout << y.data_ << std::endl;
}