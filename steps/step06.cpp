#include <cmath>
#include <iostream>
#include <memory>

#include "NumCpp.hpp"

class Variable {
 public:
  nc::NdArray<double> data_;
  nc::NdArray<double> grad_;
  explicit Variable(const nc::NdArray<double>& data);
};

class Function {
 public:
  std::unique_ptr<Variable> input_ptr;
  virtual ~Function() {}
  Variable operator()(const Variable& input) {
    const auto& x = input.data_;
    const auto& y = this->forward(x);
    auto output = Variable(y);
    input_ptr = std::make_unique<Variable>(input);
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
    const auto& x = this->input_ptr->data_;
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
    const auto& x = this->input_ptr->data_;
    const auto& gx = exp(x) * gy;
    return gx;
  }
};

Variable::Variable(const nc::NdArray<double>& data) : data_(data) {}

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
  y.grad_ = loss;
  b.grad_ = C.backward(y.grad_);
  a.grad_ = B.backward(b.grad_);
  x.grad_ = A.backward(a.grad_);
  std::cout << x.grad_ << std::endl;
}