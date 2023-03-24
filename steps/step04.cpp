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
  Variable operator()(const Variable& input) {
    auto x = input.data_;
    auto y = this->forward(x);
    auto output = Variable(y);
    return output;
  }

  virtual nc::NdArray<double> forward(const nc::NdArray<double>& x) = 0;
};

class Square : public Function {
 public:
  nc::NdArray<double> forward(const nc::NdArray<double>& x) override {
    return x * x;
  }
};

class Exp : public Function {
 public:
  nc::NdArray<double> forward(const nc::NdArray<double>& x) override {
    return exp(x);
  }
};

Variable::Variable(const nc::NdArray<double>& data) : data_(data) {}

Variable f(const Variable& x) {
  Square A;
  Exp B;
  Square C;
  return C(B(A(x)));
}

nc::NdArray<double> numerical_diff(std::function<Variable(Variable)> f,
                                   const Variable& x, double eps = 1e-04) {
  auto x0 = Variable(x.data_ - eps);
  auto x1 = Variable(x.data_ + eps);
  auto y0 = f(x0);
  auto y1 = f(x1);
  return (y1.data_ - y0.data_) / (2 * eps);
}

int main() {
  nc::NdArray<double> data({0.5});
  auto x = Variable(data);

  auto dy = numerical_diff(f, x);
  std::cout << dy << std::endl;
}