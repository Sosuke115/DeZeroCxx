#include "core_simple.h"

bool Config::enable_backprop = true;

Variable::Variable(const NdArrPtr& data, const std::string& name)
    : data(data), name(name){};

void Variable::set_creator(FuncPtr creator) {
  creator_ptr = creator;
  this->generation = creator->generation + 1;
}
void Variable::backward(const bool retain_grad) {
  if (!this->grad) {
    // 勾配の初期値を設定
    auto tmp = nc::ones_like<double>(*this->data);
    this->grad = as_array(tmp);
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
    std::vector<NdArrPtr> gys;
    for (auto& output : f->outputs_) {
      gys.push_back(output.lock()->grad);
    }
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
        f->inputs_[i]->grad = as_array(*f->inputs_[i]->grad + *gxs[i]);
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
void Variable::cleargrad() { grad = nullptr; }
nc::Shape Variable::shape() { return this->data->shape(); };
nc::uint32 Variable::size() { return this->data->size(); }

std::vector<NdArrPtr> Add::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  std::vector<NdArrPtr> ys = {as_array((*xs[0]) + (*xs[1]))};
  return ys;
}
std::vector<NdArrPtr> Add::backward(const std::vector<NdArrPtr>& gy) {
  std::vector<NdArrPtr> gx = {gy[0], gy[0]};
  return gx;
}

std::vector<NdArrPtr> Mul::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  std::vector<NdArrPtr> ys = {as_array((*xs[0]) * (*xs[1]))};
  return ys;
}
std::vector<NdArrPtr> Mul::backward(const std::vector<NdArrPtr>& gy) {
  assert(gy.size() == 1);
  const auto& x0 = this->inputs_[0]->data;
  const auto& x1 = this->inputs_[1]->data;
  std::vector<NdArrPtr> gx = {as_array((*x1) * (*gy[0])),
                              as_array((*x0) * (*gy[0]))};
  return gx;
}

std::vector<NdArrPtr> Neg::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  std::vector<NdArrPtr> ys = {as_array(-*xs[0])};
  return ys;
}
std::vector<NdArrPtr> Neg::backward(const std::vector<NdArrPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<NdArrPtr> gx = {as_array(-*gy[0])};
  return gx;
}

std::vector<NdArrPtr> Sub::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  auto res = as_array(*xs[0] - *xs[1]);
  std::vector<NdArrPtr> ys = {res};
  return ys;
}
std::vector<NdArrPtr> Sub::backward(const std::vector<NdArrPtr>& gy) {
  assert(gy.size() == 1 ||
         (std::cerr << "Error: gy.size() = " << gy.size() << "\n", false));
  std::vector<NdArrPtr> gx = {as_array(*gy[0]), as_array(-*gy[0])};
  return gx;
}

std::vector<NdArrPtr> Div::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  auto res = as_array(*xs[0] / *xs[1]);
  std::vector<NdArrPtr> ys = {res};
  return ys;
}
std::vector<NdArrPtr> Div::backward(const std::vector<NdArrPtr>& gy) {
  assert(gy.size() == 1);
  const auto& x0 = this->inputs_[0]->data;
  const auto& x1 = this->inputs_[1]->data;
  std::vector<NdArrPtr> gx = {as_array(*gy[0] / (*x1)),
                              as_array(*gy[0] * ((*x0) / (*x1) * (*x1)))};
  return gx;
}

Pow::Pow(int c) : c(c) {}
std::vector<NdArrPtr> Pow::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  return {as_array(nc::power(*xs[0], this->c))};
}
std::vector<NdArrPtr> Pow::backward(const std::vector<NdArrPtr>& gy) {
  assert(gy.size() == 1);
  const auto& x0 = this->inputs_[0]->data;
  std::vector<NdArrPtr> gx = {as_array(static_cast<double>(this->c) *
                                       nc::power(*x0, this->c - 1) * (*gy[0]))};
  return gx;
}