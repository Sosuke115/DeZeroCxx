#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>

#include "NumCpp.hpp"

class Function;
class Variable;

using FuncPtr = std::shared_ptr<Function>;
using VarPtr = std::shared_ptr<Variable>;
using VarPtrW = std::weak_ptr<Variable>;

class Variable {
 public:
  nc::NdArray<double> data;
  nc::NdArray<double> grad;
  FuncPtr creator_ptr;
  explicit Variable(const nc::NdArray<double>& data);
  void set_creator(FuncPtr creator);
  void backward();
};

class Function : public std::enable_shared_from_this<Function> {
 public:
  VarPtr input_ptr;
  // 循環参照を避けるためにstd::weak_ptrで管理
  VarPtrW output_ptr;

  virtual ~Function() {}
  VarPtr operator()(const VarPtr& input) {
    const auto& x = input->data;
    const auto& y = this->forward(x);
    input_ptr = input;
    // 直接std::weak_ptrに渡すと.lock()が必要でset_creatorでヌルポエラー
    VarPtr output_ptr_tmp = std::make_shared<Variable>(y);
    output_ptr_tmp->set_creator(shared_from_this());
    output_ptr = output_ptr_tmp;
    return output_ptr_tmp;
  }

  virtual nc::NdArray<double> forward(const nc::NdArray<double>& x) = 0;
  virtual nc::NdArray<double> backward(const nc::NdArray<double>& gy) = 0;
};

Variable::Variable(const nc::NdArray<double>& data) : data(data){};
void Variable::set_creator(FuncPtr creator) { creator_ptr = creator; }
void Variable::backward() {
  if (grad.data() == 0) {
    grad = nc::ones_like<double>(data);
  }
  std::deque<FuncPtr> funcs = {this->creator_ptr};
  while (!funcs.empty()) {
    FuncPtr f = funcs.front();
    funcs.pop_front();
    f->input_ptr->grad = f->backward(f->output_ptr.lock()->grad);
    if (f->input_ptr->creator_ptr) {
      funcs.push_back(f->input_ptr->creator_ptr);
    }
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

VarPtr square(VarPtr input) {
  auto f = std::make_shared<Square>();
  return (*f)(input);
}

VarPtr exp(VarPtr input) {
  auto f = std::make_shared<Exp>();
  return (*f)(input);
}

int main() {
  nc::NdArray<double> data({0.5});
  auto x = std::make_shared<Variable>(data);
  auto y = square(exp(square(x)));
  y->backward();
  std::cout << x->grad << std::endl;
}