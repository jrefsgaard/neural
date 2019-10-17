#include "neural/Layer.h"
#include <cstdlib>

using namespace arma;
using namespace std;

namespace neural {

Layer::Layer(int size, int nInputs)
{
  //We initialise with random weights and zero bias.
  arma_rng::set_seed_random();
  weights.randn(size,nInputs);
  weights / sqrt(nInputs);
  biases.randn(size);
  //weights.fill(0.1);
  //biases.fill(0.12);
}

Layer::~Layer() {}

void Layer::SetWeights(mat &newWeights)
{
  if(size(newWeights) == size(weights)) weights = newWeights;
  else{
    cout << "Layer::SetWeights(): Size mismatch, old size = " << size(weights);
    cout << ",  given size = " << size(newWeights) << endl;
    exit(EXIT_FAILURE);
  }
}

void Layer::SetBiases(vec &newBiases)
{
  if(size(newBiases) == size(biases)) biases = newBiases;
  else{
    cout << "Layer::SetBiases(): Size mismatch, old size = " << size(biases);
    cout << ",  given size = " << size(newBiases) << endl;
    exit(EXIT_FAILURE);
  }
}

void Layer::IncrementWeights(mat &dw)
{
  weights += dw;
}

void Layer::IncrementBiases(vec &db)
{
  biases += db;
}

const mat & Layer::GetWeights()
{
  return weights;
}

const vec & Layer::GetBiases()
{
  return biases;
}

int Layer::GetNNeurons()
{
  return weights.n_rows;
}

int Layer::GetNInputs()
{
  return weights.n_cols;
}
}
