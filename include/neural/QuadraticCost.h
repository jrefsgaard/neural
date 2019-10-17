#ifndef QUADRATIC_COST_H
#define QUADRATIC_COST_H
#include <armadillo>
#include "CostFunction.h"
#include "Network.h"

namespace neural {

class QuadraticCost : public CostFunction {
  public:
    QuadraticCost() = default;
    ~QuadraticCost() = default;

    using CostFunction::Eval;

    virtual double Eval(arma::vec &a, arma::vec &y);
    virtual arma::vec Delta(arma::vec &a, arma::vec &y, Network &network);
};
}
#endif
