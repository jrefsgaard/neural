#include "neural/SigmoidFunction.h"
//#include <cmath>

using namespace arma;

namespace neural {

vec SigmoidFunction::Eval(vec x)
{
  return 1./(1+exp(-x));
}

vec SigmoidFunction::Deriv(vec x)
{
  vec sigmoid = Eval(x);
  return sigmoid % (1. - sigmoid);
}
}
