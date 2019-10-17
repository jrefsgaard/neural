#include "neural/TanhFunction.h"

using namespace arma;

namespace neural {

vec TanhFunction::Eval(vec x)
{
  return tanh(x);
}

vec TanhFunction::Deriv(vec x)
{
  return 1. - square(tanh(x));
}
}
