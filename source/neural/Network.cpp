#include "neural/Network.h"

using namespace std;
using namespace arma;

namespace neural {

Network::Network(vector<int> layerSizes, string activationType)
{
  vec a0; a0.zeros(layerSizes.at(0));
  a.push_back(a0);
  for(int i=1; i<layerSizes.size(); i++){
    int size = layerSizes.at(i);
    int nInputs = layerSizes.at(i-1);
    Layer li(size,nInputs);
    layers.push_back(li);
    vec zl; zl.zeros(size);
    z.push_back(zl);
    vec al; al.zeros(size);
    a.push_back(al);
    unique_ptr<ActivationFunction> fi = unique_ptr<ActivationFunction>(ActivationFunction::Create(activationType));
    activationFunctions.push_back(move(fi));
  }
  
}

vec Network::FeedForward(vec &input)
{
  a.at(0) = input;
  for(int i=0; i<layers.size(); i++){
    Layer &l = layers.at(i);
    z.at(i) = l.GetWeights() * a.at(i) + l.GetBiases();
    a.at(i+1) = activationFunctions.at(i)->Eval(z.at(i));
  }
  return a.at(layers.size());
}

vec & Network::Z(int l)
{
  return z.at(l-1); //Correct for the fact that we don't store z for the 0th layer
}

vec & Network::A(int l)
{
  return a.at(l);
}

int Network::L()
{
  return layers.size() + 1;
}

void Network::SetActivationFunction(int l, string activationType)
{
  unique_ptr<ActivationFunction> f = unique_ptr<ActivationFunction>(ActivationFunction::Create(activationType));
  activationFunctions.at(l-1) = move(f);
}

ActivationFunction & Network::GetActivationFunction(int l)
{
  return *(activationFunctions.at(l-1).get());
}

Layer & Network::GetLayer(int l)
{
  return layers.at(l-1);
}
}
