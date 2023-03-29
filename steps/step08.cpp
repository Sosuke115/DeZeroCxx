#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>

#include "NumCpp.hpp"

class Function;

class Variable {
 public:
  nc::NdArray<double> data;
  nc::NdArray<double> grad;
  Function* creator_ptr = nullptr;
  explicit Variable(const nc::NdArray<double>& data);
  void set_creator(Function* creator);
  void backward();
};

class Function {
 public:
  Variable* input_ptr;
  Variable* output_ptr;
  virtual ~Function() {}
  Variable operator()(Variable& input) {
    const auto& x = input.data;
    const auto& y = this->forward(x);
    auto output = Variable(y);
    output.set_creator(this);
    input_ptr = &input;
    output_ptr = &output;
    return output;
  }

  virtual nc::NdArray<double> forward(const nc::NdArray<double>& x) = 0;
  virtual nc::NdArray<double> backward(const nc::NdArray<double>& gy) = 0;
};

Variable::Variable(const nc::NdArray<double>& data) : data(data){};
void Variable::set_creator(Function* creator) { creator_ptr = creator; }
void Variable::backward() {
  std::deque<Function*> funcs = {this->creator_ptr};
  while (!funcs.empty()) {
    Function* f = funcs.back();
    f->input_ptr->grad = f->backward(f->output_ptr->grad);
    if (f->input_ptr->creator_ptr != nullptr) {
      funcs.push_back(f->input_ptr->creator_ptr);
    }
    funcs.pop_front();
  }
}

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

int main() {
  auto A = Square();
  auto B = Exp();
  auto C = Square();
  nc::NdArray<double> data({0.5});
  auto x = Variable(data);

  auto a = A(x);
  auto b = B(a);
  auto y = C(b);

  assert(y.creator_ptr == &C);
  assert(y.creator_ptr->input_ptr == &b);

  nc::NdArray<double> loss({1.0});
  y.grad = loss;
  y.backward();
  std::cout << x.grad << std::endl;
}