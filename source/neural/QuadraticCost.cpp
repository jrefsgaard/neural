#include "neural/QuadraticCost.h"
#include <cmath>

using namespace arma;
using namespace std;

namespace neural {

double QuadraticCost::Eval(vec &a, vec &y)
{
  double n = norm(y-a,2); //Euclidian norm.
  return 0.5 * pow(n,2);  //Norm squared.
}

vec QuadraticCost::Delta(vec &a, vec &y, Network &network)
{
  int L = network.L();
  vec zL = network.Z(L-1);
  vec sigmap = network.GetActivationFunction(L-1).Deriv(zL);
  return (a-y) % sigmap;
}
}
