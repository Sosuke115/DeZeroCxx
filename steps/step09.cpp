#include "step09.h"

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

nc::NdArray<double> Square::forward(const nc::NdArray<double>& x) {
  return x * x;
}
nc::NdArray<double> Square::backward(const nc::NdArray<double>& gy) {
  const auto& x = this->input_ptr->data;
  const auto& gx = 2.0 * x * gy;
  return gx;
}

nc::NdArray<double> Exp::forward(const nc::NdArray<double>& x) {
  return exp(x);
}
nc::NdArray<double> Exp::backward(const nc::NdArray<double>& gy) {
  const auto& x = this->input_ptr->data;
  const auto& gx = exp(x) * gy;
  return gx;
}

int main() {
  nc::NdArray<double> data({0.5});
  auto x = std::make_shared<Variable>(data);
  auto y = square(exp(square(x)));
  y->backward();
  std::cout << x->grad << std::endl;
}