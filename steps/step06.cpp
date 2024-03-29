#include <cmath>
#include <iostream>
#include <memory>

#include "NumCpp.hpp"

class Variable {
 public:
  nc::NdArray<double> data;
  nc::NdArray<double> grad;
  explicit Variable(const nc::NdArray<double>& data);
};

class Function {
 public:
  Variable* input_ptr;
  virtual ~Function() {}
  Variable operator()(Variable& input) {
    const auto& x = input.data;
    const auto& y = this->forward(x);
    auto output = Variable(y);
    input_ptr = &input;
    return output;
  }

  virtual nc::NdArray<double> forward(const nc::NdArray<double>& x) = 0;
  virtual nc::NdArray<double> backward(const nc::NdArray<double>& gy) = 0;
};

class Square : public Function {
 public:
  nc::NdArray<double> forward(const nc::NdArray<double>& x) override {
    return x * x;
  }
  nc::NdArray<double> backward(const nc::NdArray<double>& gy) override {
    const auto& x = this->input_ptr->data;
    const auto& gx = 2.0 * x * gy;
    return gx;
  }
};

class Exp : public Function {
 public:
  nc::NdArray<double> forward(const nc::NdArray<double>& x) override {
    return exp(x);
  }
  nc::NdArray<double> backward(const nc::NdArray<double>& gy) override {
    const auto& x = this->input_ptr->data;
    const auto& gx = exp(x) * gy;
    return gx;
  }
};

Variable::Variable(const nc::NdArray<double>& data) : data(data) {}

int main() {
  auto A = Square();
  auto B = Exp();
  auto C = Square();
  nc::NdArray<double> data({0.5});
  auto x = Variable(data);

  auto a = A(x);
  auto b = B(a);
  auto y = C(b);

  nc::NdArray<double> loss({1.0});
  y.grad = loss;
  b.grad = C.backward(y.grad);
  a.grad = B.backward(b.grad);
  x.grad = A.backward(a.grad);
  std::cout << x.grad << std::endl;
}