#ifndef LAYER_H
#define LAYER_H
#include <armadillo>

namespace neural {

class Layer {
  private:
    arma::mat weights;
    arma::vec biases;

  public:
    Layer(int size, int nInputs);
    ~Layer();

    void SetWeights(arma::mat &newWeights);
    void SetBiases(arma::vec &newBiases);

    void IncrementWeights(arma::mat &dw);
    void IncrementBiases(arma::vec &db);

    const arma::mat & GetWeights();
    const arma::vec & GetBiases();
    int GetNNeurons();
    int GetNInputs();
};
}
#endif
