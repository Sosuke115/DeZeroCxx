#ifndef FUNCTIONS_
#define FUNCTIONS_

#include <memory>

#include "NumCpp.hpp"
#include "core.h"
#include "utils.h"
class Function;
class Variable;

using FuncPtr = std::shared_ptr<Function>;
using VarPtr = std::shared_ptr<Variable>;
using VarPtrW = std::weak_ptr<Variable>;
using NdArrPtr = std::shared_ptr<nc::NdArray<double>>;

namespace F {
class Sin : public Function {
 public:
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Cos : public Function {
 public:
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Tanh : public Function {
 public:
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Reshape : public Function {
 public:
  const nc::Shape shape;
  nc::Shape x_shape;
  explicit Reshape(const nc::Shape& shape);
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class Transpose : public Function {
 public:
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class BroadcastTo : public Function {
 public:
  const nc::Shape shape;
  nc::Shape x_shape;
  explicit BroadcastTo(const nc::Shape& shape);
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class SumTo : public Function {
 public:
  const nc::Shape shape;
  nc::Shape x_shape;
  explicit SumTo(const nc::Shape& shape);
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

// numcppのndarrayは2Dなのでkeepdims不要
class Sum : public Function {
 public:
  nc::Axis axis;
  nc::Shape x_shape;
  explicit Sum(nc::Axis axis);
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

class MatMul : public Function {
 public:
  std::vector<NdArrPtr> forward(const std::vector<NdArrPtr>& xs) override;
  std::vector<VarPtr> backward(const std::vector<VarPtr>& gy) override;
};

inline VarPtr sin(VarPtr x) {
  auto f = std::make_shared<Sin>();
  return (*f)(x)[0];
}

inline VarPtr cos(VarPtr x) {
  auto f = std::make_shared<Cos>();
  return (*f)(x)[0];
}

inline VarPtr tanh(VarPtr x) {
  auto f = std::make_shared<Tanh>();
  return (*f)(x)[0];
}

inline VarPtr reshape(VarPtr x, const nc::Shape& shape) {
  if (x->shape() == shape) {
    return as_variable(x);
  }
  auto f = std::make_shared<Reshape>(shape);
  return (*f)(x)[0];
}

inline VarPtr transpose(VarPtr x) {
  auto f = std::make_shared<Transpose>();
  return (*f)(x)[0];
}
inline VarPtr broadcast_to(VarPtr x, const nc::Shape& shape) {
  if (x->shape() == shape) {
    return as_variable(x);
  }
  auto f = std::make_shared<BroadcastTo>(shape);
  return (*f)(x)[0];
}
inline VarPtr sum_to(VarPtr x, const nc::Shape& shape) {
  if (x->shape() == shape) {
    return as_variable(x);
  }
  auto f = std::make_shared<SumTo>(shape);
  return (*f)(x)[0];
}

inline VarPtr sum(VarPtr x, nc::Axis axis = nc::Axis::NONE) {
  auto f = std::make_shared<Sum>(axis);
  return (*f)(x)[0];
}

inline VarPtr matmul(VarPtr x0, VarPtr x1) {
  auto f = std::make_shared<MatMul>();
  return (*f)(x0, x1)[0];
}

}  // namespace F
#endif