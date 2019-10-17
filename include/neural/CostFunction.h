#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H
#include <string>
#include <memory>
#include <armadillo>
#include "Network.h"
#include "DataSet.h"

namespace neural {

class CostFunction {
  public:
    CostFunction() = default;
    virtual ~CostFunction() = default;

    /**
    * Factory method for the various cost functions. Supported types are:
    *  -quadratic
    *  -crossentropy
    */
    static CostFunction * Create(std::string type);

    /**
    * Vector a is the resulting output from the network and y is the desired
    * output.
    */
    virtual double Eval(arma::vec &a, arma::vec &y) = 0;

    /**
    * Evaluate the cost-function averaged over the provided dataset.
    */
    virtual double Eval(Network &network, std::shared_ptr<DataSet> data);

    /**
    * Vector a is the resulting output from the network and y is the desired
    * output. Returns the 'error' (delta_L) of the output layer.
    */
    virtual arma::vec Delta(arma::vec &a, arma::vec &y, Network &network) = 0;
};
}
#endif
