#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include <memory>
#include <string>
#include <armadillo>
#include "neural/Layer.h"
#include "neural/ActivationFunction.h"

namespace neural {

class Network {
  private:
    std::vector<Layer> layers;
    std::vector<std::unique_ptr<ActivationFunction>> activationFunctions;
    std::vector<arma::vec> z; //z(l) = w(l)*a(l-1)+b(l).
    std::vector<arma::vec> a; //a(l) = sigma(z(l)).

  public:
    Network(std::vector<int> layerSizes, std::string activationType = "sigmoid");
    ~Network() = default;

    arma::vec FeedForward(arma::vec &input);

    /**
    * Returns w(l)*a(l-1)+b(l) calculated during the latest FeedForward-call.
    * Only produces valid output for 0 < l < L.
    */
    arma::vec & Z(int l);

    /**
    * Returns sigma(z(l)) calculated during the latest FeedForward-call.
    * Produces valid output for 0 <= l < L.
    */
    arma::vec & A(int l);

    /**
    * Get number of layers in the network (including input layer).
    */
    int L();

    void SetActivationFunction(int l, std::string activationType);

    /**
    * Get the activation function for layer l. Since the input layer has no
    * associated activation function it should only be called with 0 < l < L.
    */
    ActivationFunction & GetActivationFunction(int l);

    /**
    * Returns a reference to layer no. l. Produces valid output for 0 < l < L.
    */
    Layer & GetLayer(int l);
};
}
#endif
