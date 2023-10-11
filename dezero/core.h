#ifndef CORE_
#define CORE_

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "NumCpp.hpp"

class Function;
class Variable;

using FuncPtr = std::shared_ptr<Function>;
using VarPtr = std::shared_ptr<Variable>;
using VarPtrW = std::weak_ptr<Variable>;
using NdArrPtr = std::shared_ptr<nc::NdArray<double>>;

class Config {
 public:
  static bool enable_backprop;
};

// RAIIパターン (Resource Acquisition Is Initialization)
// https://qiita.com/wx257osn2/items/e2e3bcbfdd8bd02872aa
class UsingConfig {
 public:
  UsingConfig(const std::string& name, bool value)
      : name_(name), old_value_(Config::enable_backprop) {
    if (name == "enable_backprop") {
      Config::enable_backprop = value;
    }
  }

  ~UsingConfig() {
    if (name_ == "enable_backprop") {
      Config::enable_backprop = old_value_;
    }
  }

 private:
  std::string name_;
  bool old_value_;
};

inline UsingConfig no_grad() { return UsingConfig("enable_backprop", false); }

inline NdArrPtr as_array(const NdArrPtr& obj) { return obj; }
inline NdArrPtr as_array(const nc::NdArray<double>& obj) {
  return std::make_shared<nc::NdArray<double>>(obj);
}
inline NdArrPtr as_array(double obj) {
  nc::NdArray<double> data({obj});
  return std::make_shared<nc::NdArray<double>>(data);
}
inline NdArrPtr as_array(int obj) {
  nc::NdArray<double> data({static_cast<double>(obj)});
  return std::make_shared<nc::NdArray<double>>(data);
}
inline NdArrPtr as_array(std::initializer_list<double> obj) {
  return std::make_shared<nc::NdArray<double>>(obj);
}
template <typename T>
inline NdArrPtr as_array(const T&) {
  throw std::invalid_argument("Unsupported type passed to as_array");
}

inline VarPtr as_variable(const VarPtr& obj) { return obj; }
inline VarPtr as_variable(const NdArrPtr& obj) {
  return std::make_shared<Variable>(obj);
}
template <typename T>
inline VarPtr as_variable(const T&) {
  throw std::invalid_argument("Unsupported type passed to as_variable");
}

class Variable : public std::enable_shared_from_this<Variable> {
 public:
  NdArrPtr data;
  std::string name;
  VarPtr grad;
  FuncPtr creator_ptr;
  explicit Variable(const NdArrPtr& data, const std::string& name = "");
  void set_creator(FuncPtr creator);
  void backward(const bool retain_grad = true, const bool create_graph = false);
  void cleargrad();
  VarPtr reshape(const nc::Shape& shape);
  VarPtr transpose();
  VarPtr T();
  int generation = 0;

  nc::Shape shape();
  nc::uint32 size();

  VarPtr sum(nc::Axis axis = nc::Axis::NONE);
};

class Function : public std::enable_shared_from_this<Function> {
 public:
  std::vector<VarPtr> inputs_;
  // 循環参照を避けるためにstd::weak_ptrで管理
  std::vector<VarPtrW> outputs_;
  int generation = 0;

  virtual ~Function(){};

  template <typename... Args>
  std::vector<VarPtr> operator()(const Args&... inputs);

  // 可変長テンプレートと純粋仮想関数は両立できない
  virtual std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) = 0;
  virtual std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) = 0;
};

class Add : public Function {
 public:
  nc::Shape x0_shape;
  nc::Shape x1_shape;
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Mul : public Function {
 public:
  nc::Shape x0_shape;
  nc::Shape x1_shape;
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Neg : public Function {
 public:
  nc::Shape x0_shape;
  nc::Shape x1_shape;
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Sub : public Function {
 public:
  nc::Shape x0_shape;
  nc::Shape x1_shape;
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Div : public Function {
 public:
  nc::Shape x0_shape;
  nc::Shape x1_shape;
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Pow : public Function {
 public:
  int c;
  explicit Pow(int c);
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

inline VarPtr add(VarPtr x0, VarPtr x1) {
  auto f = std::make_shared<Add>();
  return (*f)(x0, x1)[0];
}

template <typename T>
inline VarPtr add(VarPtr x0, T x1) {
  auto f = std::make_shared<Add>();
  return (*f)(x0, as_array(x1))[0];
}

inline VarPtr mul(VarPtr x0, VarPtr x1) {
  auto f = std::make_shared<Mul>();
  return (*f)(x0, x1)[0];
}

inline VarPtr neg(VarPtr x0) {
  auto f = std::make_shared<Neg>();
  return (*f)(x0)[0];
}

inline VarPtr sub(VarPtr x0, VarPtr x1) {
  auto f = std::make_shared<Sub>();
  return (*f)(x0, x1)[0];
}

inline VarPtr div(VarPtr x0, VarPtr x1) {
  auto f = std::make_shared<Div>();
  return (*f)(x0, x1)[0];
}

inline VarPtr pow(VarPtr x0, int c) {
  auto f = std::make_shared<Pow>(c);
  return (*f)(x0)[0];
}

// 演算子オーバーロード
// テンプレートの実装はヘッダファイルに書く必要がある:
// https://pknight.hatenablog.com/entry/20090826/1251303641
// 可読性重視で演算子オーバーロードは全てヘッダファイルに書く

template <typename... Args>
inline std::vector<VarPtr> Function::operator()(const Args&... inputs) {
  // パック展開
  std::vector<NdArrPtr> xs{inputs->data...};
  const auto& ys = this->forward(xs);

  inputs_ = {inputs...};
  std::vector<VarPtr> outputs;
  for (const auto& y : ys) {
    auto output = as_variable(y);
    outputs.push_back(output);
  }

  if (Config::enable_backprop) {
    // 世代の設定
    for (const auto& input : inputs_) {
      generation = std::max(generation, input->generation);
    }
    // つながりの設定
    for (const auto& output : outputs) {
      output->set_creator(shared_from_this());
    }
  }

  outputs_.reserve(outputs.size());
  std::copy(outputs.begin(), outputs.end(), std::back_inserter(outputs_));
  return outputs;
}

inline std::ostream& operator<<(std::ostream& os, const VarPtr& var) {
  if (!(var->data)) {
    return os;
  }
  auto p = var->data->str();

  // 文字列の置換を行う関数
  auto replace = [](std::string& s, const std::string& target,
                    const std::string& replacement) {
    if (!target.empty()) {
      std::string::size_type pos = 0;
      while ((pos = s.find(target, pos)) != std::string::npos) {
        s.replace(pos, target.length(), replacement);
        pos += replacement.length();
      }
    }
  };

  replace(p, "\n", "\n         ");

  os << "variable(" << p << ")";
  return os;
};

inline VarPtr operator+(const VarPtr& lv, const VarPtr& rv) {
  return add(lv, rv);
}
template <typename T>
inline VarPtr operator+(const VarPtr& lv, T rv) {
  return add(lv, as_variable(as_array(rv)));
}
template <typename T>
inline VarPtr operator+(T lv, const VarPtr& rv) {
  return add(as_variable(as_array(lv)), rv);
}
inline VarPtr operator-(const VarPtr& lv, const VarPtr& rv) {
  return sub(lv, rv);
}
template <typename T>
inline VarPtr operator-(const VarPtr& lv, T rv) {
  return sub(lv, as_variable(as_array(rv)));
}
template <typename T>
inline VarPtr operator-(T lv, const VarPtr& rv) {
  return sub(as_variable(as_array(lv)), rv);
}
inline VarPtr operator/(const VarPtr& lv, const VarPtr& rv) {
  return div(lv, rv);
}
template <typename T>
inline VarPtr operator/(const VarPtr& lv, T rv) {
  return div(lv, as_variable(as_array(rv)));
}
template <typename T>
inline VarPtr operator/(T lv, const VarPtr& rv) {
  return div(as_variable(as_array(lv)), rv);
}
inline VarPtr operator*(const VarPtr& lv, const VarPtr& rv) {
  return mul(lv, rv);
}
template <typename T>
inline VarPtr operator*(const VarPtr& lv, T rv) {
  return mul(lv, as_variable(as_array(rv)));
}
template <typename T>
inline VarPtr operator*(T lv, const VarPtr& rv) {
  return mul(as_variable(as_array(lv)), rv);
}
inline VarPtr operator-(const VarPtr& v) { return neg(v); }
#endif