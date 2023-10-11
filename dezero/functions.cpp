#include "functions.h"

namespace F {
std::vector<NdArrPtr> Sin::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  std::vector<NdArrPtr> ys = {as_array(nc::sin(*xs[0]))};
  return ys;
}
std::vector<VarPtr> Sin::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {gy[0] * cos(this->inputs_[0])};
  return gx;
}

std::vector<NdArrPtr> Cos::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  std::vector<NdArrPtr> ys = {as_array(nc::cos(*xs[0]))};
  return ys;
}
std::vector<VarPtr> Cos::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {gy[0] * -sin(this->inputs_[0])};
  return gx;
}

std::vector<NdArrPtr> Tanh::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  std::vector<NdArrPtr> ys = {as_array(nc::tanh(*xs[0]))};
  return ys;
}
std::vector<VarPtr> Tanh::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {
      gy[0] * (1.0 - this->outputs_[0].lock() * this->outputs_[0].lock())};
  return gx;
}

Reshape::Reshape(const nc::Shape& shape) : shape(shape) {}
std::vector<NdArrPtr> Reshape::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  this->x_shape = xs[0]->shape();
  std::vector<NdArrPtr> ys = {as_array(xs[0]->reshape(this->shape))};
  return ys;
}
std::vector<VarPtr> Reshape::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {F::reshape(gy[0], this->x_shape)};
  return gx;
}

std::vector<NdArrPtr> Transpose::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  std::vector<NdArrPtr> ys = {as_array(nc::transpose(*xs[0]))};
  return ys;
}
std::vector<VarPtr> Transpose::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {transpose(gy[0])};
  return gx;
}

BroadcastTo::BroadcastTo(const nc::Shape& shape) : shape(shape) {}
std::vector<NdArrPtr> BroadcastTo::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  this->x_shape = xs[0]->shape();
  std::vector<NdArrPtr> ys = {
      as_array(utils::broadcast_to(*xs[0], nc::Shape(this->shape)))};
  return ys;
}
std::vector<VarPtr> BroadcastTo::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {sum_to(gy[0], nc::Shape(this->x_shape))};
  return gx;
}

SumTo::SumTo(const nc::Shape& shape) : shape(shape) {}
std::vector<NdArrPtr> SumTo::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  this->x_shape = xs[0]->shape();
  std::vector<NdArrPtr> ys = {
      as_array(utils::sum_to(*xs[0], nc::Shape(this->shape)))};
  return ys;
}
std::vector<VarPtr> SumTo::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  std::vector<VarPtr> gx = {broadcast_to(gy[0], this->x_shape)};
  return gx;
}

Sum::Sum(nc::Axis axis) : axis(axis) {}
std::vector<NdArrPtr> Sum::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 1);
  this->x_shape = xs[0]->shape();
  // 現状の実装だと軸に限らずshapeが(1, N)になる点に注意
  std::vector<NdArrPtr> ys = {as_array(nc::sum(*xs[0], this->axis))};
  return ys;
}
std::vector<VarPtr> Sum::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  // numcppのndarrayは2Dなのでreshape不要
  std::vector<VarPtr> gx = {broadcast_to(gy[0], this->x_shape)};
  return gx;
}

std::vector<NdArrPtr> MatMul::forward(const std::vector<NdArrPtr>& xs) {
  assert(xs.size() == 2);
  std::vector<NdArrPtr> ys = {as_array((*xs[0]).dot(*xs[1]))};
  return ys;
}
std::vector<VarPtr> MatMul::backward(const std::vector<VarPtr>& gy) {
  assert(gy.size() == 1);
  const auto& x0 = this->inputs_[0];
  const auto& W = this->inputs_[1];
  auto gx0 = F::matmul(gy[0], W->T());
  auto gW = F::matmul(x0->T(), gy[0]);
  std::vector<VarPtr> gx = {gx0, gW};
  return gx;
}

}  // namespace F