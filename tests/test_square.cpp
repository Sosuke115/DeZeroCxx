#include <gtest/gtest.h>

#include "NumCpp.hpp"
#include "step09.h"

nc::NdArray<double> numerical_diff(std::function<VarPtr(VarPtr)> f,
                                   const VarPtr& x, double eps = 1e-04) {
  auto x0 = std::make_shared<Variable>(x->data - eps);
  auto x1 = std::make_shared<Variable>(x->data + eps);
  auto y0 = f(x0);
  auto y1 = f(x1);
  return (y1->data - y0->data) / (2 * eps);
}

class SquareTest : public ::testing::Test {
 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

// gradient checking
TEST_F(SquareTest, gradientCheckTest) {
  nc::NdArray<double> data({0.5});
  auto x = std::make_shared<Variable>(data);
  auto y = square(x);
  y->backward();
  auto num_x_grad = numerical_diff(square, x);
  auto flg = nc::allclose(x->grad, num_x_grad);
  EXPECT_TRUE(flg);
}