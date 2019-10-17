#include "neural/LogLikelihood.h"
#include "neural/SoftmaxFunction.h"
#include <cmath>
#include <typeinfo>

using namespace arma;
using namespace std;

namespace neural {

double LogLikelihood::Eval(vec &a, vec &y)
{
  //We assume this is some kind of classification problem with exactly one
  //correct answer, i.
  int i = y.index_max();
  return -log(a(i));
}

vec LogLikelihood::Delta(vec &a, vec &y, Network &network)
{
  //This Cost-function should mostly be used in conjunction with the softmax
  //activation function in the output layer of the network.
  int L = network.L();
  if(typeid(network.GetActivationFunction(L-1)) == typeid(SoftmaxFunction)){
    return a - y;
  }
  else{
    vec zL = network.Z(L-1);
    vec sigmap = network.GetActivationFunction(L-1).Deriv(zL);
    int i = y.index_max();
    vec b;
    b.zeros(size(a));
    b(i) = -1./a(i) * sigmap(i);
    return b;
  }
}
}
