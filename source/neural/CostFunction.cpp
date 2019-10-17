#include "neural/CostFunction.h"
#include "neural/QuadraticCost.h"
#include "neural/CrossEntropy.h"
#include "neural/LogLikelihood.h"
#include <iostream>
#include <cstdlib>

using namespace arma;
using namespace std;

namespace neural {

CostFunction * CostFunction::Create(string type)
{
  if(type == "quadratic"){
    return new QuadraticCost();
  }
  else if(type == "crossentropy"){
    return new CrossEntropy();
  }
  else if(type == "loglikelihood"){
    return new LogLikelihood();
  }
  else{
    cout << "CostFunction::Create(): Unknown type: " << type << endl;
    exit(EXIT_FAILURE);
  }
}

double CostFunction::Eval(Network &network, shared_ptr<DataSet> data)
{
  int n = data->GetSize();
  double cost;
  for(int i=0; i<n; i++){
    array<vec,2> &point = data->GetDataPoint(i);
    vec &x = point[0];
    vec &y = point[1];
    vec a = network.FeedForward(x);
    cost += Eval(a,y);
  }
  cost /= n;
  return cost;
}
}
