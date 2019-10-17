#include <neural/SoftmaxFunction.h>
#include <cstdlib>

using namespace std;
using namespace arma;

namespace neural {

vec SoftmaxFunction::Eval(vec x)
{
  vec ez = exp(x);
  return ez / sum(ez);
}

vec SoftmaxFunction::Deriv(vec x)
{
  vec a = Eval(x);
  return a - square(a);
}
}
