#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H
#include <armadillo>
#include "CostFunction.h"
#include "Network.h"

namespace neural {

class CrossEntropy : public CostFunction {
  public:
    CrossEntropy() = default;
    ~CrossEntropy() = default;

    using CostFunction::Eval;

    virtual double Eval(arma::vec &a, arma::vec &y);
    virtual arma::vec Delta(arma::vec &a, arma::vec &y, Network &network);
};
}
#endif
