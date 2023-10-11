#include "core.h"

#include "functions.h"
#include "utils.h"

bool Config::enable_backprop = true;

Variable::Variable(const NdArrPtr& data, const std::string& name)
    : data(data), name(name){};

void Variable::set_creator(FuncPtr creator) {
  creator_ptr = creator;
  this->generation = creator->generation + 1;
}
void Variable::backward(const bool retain_grad, const bool create_graph) {
  if (!this->grad) {
    // 勾配の初期値を設定
    this->grad = as_variable(as_array(nc::ones_like<double>(*this->data)));
  }
  std::vector<FuncPtr> funcs = {};
  // 処理済み関数
  std::unordered_set<FuncPtr> seen_set;

  auto add_func = [&funcs, &seen_set](const FuncPtr& f) {
    // TODO: 優先度キーでの実装が効率的
    if (!seen_set.count(f)) {
      funcs.push_back(f);
      seen_set.insert(f);
      // generation昇順にソート
      std::sort(funcs.begin(), funcs.end(),
                [](const FuncPtr& lhs, const FuncPtr& rhs) {
                  return lhs->generation < rhs->generation;
                });
    }
  };

  add_func(this->creator_ptr);

  while (!funcs.empty()) {
    FuncPtr f = funcs.back();
    funcs.pop_back();
    std::vector<VarPtr> gys;
    for (auto& output : f->outputs_) {
      gys.push_back(output.lock()->grad);
    }
    {
      // with_backprop_cfgが生きている間（このスコープを抜けるまで）Config::enable_backprop
      // = create_graphとなる（Pythonで言うところのwith構文の代替）
      UsingConfig with_backprop_cfg("enable_backprop", create_graph);

      const auto& gxs = f->backward(gys);

      // 入力データと勾配のサイズは等しい
      assert(gxs.size() == f->inputs_.size());

      for (int i = 0; i < f->inputs_.size(); i++) {
        if (!f->inputs_[i]->grad) {
          f->inputs_[i]->grad = gxs[i];
        } else {
          // 付録A参照
          // 新しくインスタンスを作成（コピー）する必要がある。インプレース演算（*f->inputs_[i]->grad
          // += *gxs[i]）にしてしまうと
          // 例えばy.gradとx.gradが同じインスタンスを参照してしまう。
          f->inputs_[i]->grad = as_variable(f->inputs_[i]->grad + gxs[i]);
        }

        if (f->inputs_[i]->creator_ptr) {
          // １つ前の関数をリストに追加
          add_func(f->inputs_[i]->creator_ptr);
        }
      }

      if (!retain_grad) {
        for (auto& output : f->outputs_) {
          output.lock()->grad = nullptr;
        }
      }
    }
  }
}
void Variable::cleargrad() { grad = nullptr; }
VarPtr Variable::reshape(const nc::Shape& shape) {
  return F::reshape(shared_from_this(), shape);
};
VarPtr Variable::transpose() { return F::transpose(shared_from_this()); };
VarPtr Variable::T() { return F::transpose(shared_from_this()); };
nc::Shape Variable::shape() { return this->data->shape(); };
nc::uint32 Variable::size() { return this->data->size(); }
VarPtr Variable::sum(nc::Axis axis) {
  return F::sum(shared_from_this(), axis);
};

std::vector<NdArrPtr> Add::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  this->x0_shape = xs[0]->shape();
  this->x1_shape = xs[1]->shape();
  std::vector<NdArrPtr> ys = {as_array((*xs[0]) + (*xs[1]))};
  return ys;
}
std::vector<VarPtr> Add::backward(const std::vector<VarPtr>& gy) {
  auto gx0 = gy[0];
  auto gx1 = gy[0];
  if (this->x0_shape != this->x1_shape) {
    gx0 = F::sum_to(gx0, this->x0_shape);
    gx1 = F::sum_to(gx1, this->x1_shape);
  }
  std::vector<VarPtr> gx = {gx0, gx1};
  return gx;
}

std::vector<NdArrPtr> Mul::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  this->x0_shape = xs[0]->shape();
  this->x1_shape = xs[1]->shape();
  std::vector<NdArrPtr> ys = {as_array((*xs[0]) * (*xs[1]))};
  return ys;
}
std::vector<VarPtr> Mul::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  const auto& x0 = this->inputs_[0]->data;
  const auto& x1 = this->inputs_[1]->data;
  auto gx0 = this->inputs_[1] * gy[0];
  auto gx1 = this->inputs_[0] * gy[0];
  if (this->x0_shape != this->x1_shape) {
    gx0 = F::sum_to(gx0, this->x0_shape);
    gx1 = F::sum_to(gx1, this->x1_shape);
  }
  std::vector<VarPtr> gx = {gx0, gx1};
  return gx;
}

std::vector<NdArrPtr> Neg::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  this->x0_shape = xs[0]->shape();
  this->x1_shape = xs[1]->shape();
  std::vector<NdArrPtr> ys = {as_array(-*xs[0])};
  return ys;
}
std::vector<VarPtr> Neg::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {-gy[0]};
  return gx;
}

std::vector<NdArrPtr> Sub::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  this->x0_shape = xs[0]->shape();
  this->x1_shape = xs[1]->shape();
  auto res = as_array(*xs[0] - *xs[1]);
  std::vector<NdArrPtr> ys = {res};
  return ys;
}
std::vector<VarPtr> Sub::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  auto gx0 = gy[0];
  auto gx1 = -gy[0];
  if (this->x0_shape != this->x1_shape) {
    gx0 = F::sum_to(gx0, this->x0_shape);
    gx1 = F::sum_to(gx1, this->x1_shape);
  }
  std::vector<VarPtr> gx = {gx0, gx1};
  return gx;
}

std::vector<NdArrPtr> Div::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  this->x0_shape = xs[0]->shape();
  this->x1_shape = xs[1]->shape();
  auto res = as_array(*xs[0] / *xs[1]);
  std::vector<NdArrPtr> ys = {res};
  return ys;
}
std::vector<VarPtr> Div::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  const auto& x0 = this->inputs_[0]->data;
  const auto& x1 = this->inputs_[1]->data;
  auto gx0 = gy[0] / this->inputs_[1];
  auto gx1 =
      gy[0] * ((this->inputs_[0]) / (this->inputs_[1]) * (this->inputs_[1]));
  if (this->x0_shape != this->x1_shape) {
    gx0 = F::sum_to(gx0, this->x0_shape);
    gx1 = F::sum_to(gx1, this->x1_shape);
  }
  std::vector<VarPtr> gx = {gx0, gx1};
  return gx;
}

Pow::Pow(int c) : c(c) {}
std::vector<NdArrPtr> Pow::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  return {as_array(nc::power(*xs[0], this->c))};
}
std::vector<VarPtr> Pow::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {static_cast<double>(this->c) *
                            pow(this->inputs_[0], this->c - 1) * (gy[0])};
  return gx;
}