#include "neural/ActivationFunction.h"
#include "neural/SigmoidFunction.h"
#include "neural/SoftmaxFunction.h"
#include "neural/TanhFunction.h"
#include "neural/RectifiedLinear.h"
#include <iostream>
#include <cstdlib>

using namespace arma;
using namespace std;

namespace neural {

ActivationFunction * ActivationFunction::Create(string type)
{
  if(type == "sigmoid"){
    return new SigmoidFunction();
  }
  else if(type == "softmax"){
    return new SoftmaxFunction();
  }
  else if(type == "tanh"){
    return new TanhFunction();
  }
  else if(type == "rectifiedlinear"){
    return new RectifiedLinear();
  }
  else{
    cout << "ActivationFunction::Create(): Unknown type: " << type << endl;
    exit(EXIT_FAILURE);
  }
}
}
