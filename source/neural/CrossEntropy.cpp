#include "neural/CrossEntropy.h"
#include "neural/SigmoidFunction.h"
#include <cmath>
#include <typeinfo>
#include <iostream>

using namespace arma;
using namespace std;

namespace neural {

double CrossEntropy::Eval(vec &a, vec &y)
{
  return -sum(y % trunc_log(a) + (1.-y) % trunc_log(1.-a));
}

vec CrossEntropy::Delta(vec &a, vec &y, Network &network)
{
  int L = network.L();
  if(typeid(network.GetActivationFunction(L-1)) == typeid(SigmoidFunction)){ 
    return a - y;
  }
  else{
    vec zL = network.Z(L-1);
    vec sigmap = network.GetActivationFunction(L-1).Deriv(zL);
    return ((a-y) / (a % (1.-a))) % sigmap;
  }
}
}
