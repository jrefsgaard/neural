#include "neural/RectifiedLinear.h"
#include <iostream>

using namespace arma;
using std::cout;
using std::endl;

namespace neural {

vec RectifiedLinear::Eval(vec x)
{
  vec a;
  a.zeros(size(x));
  return max(a,x);
}

vec RectifiedLinear::Deriv(vec x)
{
  vec a;
  a.zeros(size(x));
  return conv_to<vec>::from(x > a);
}
}
