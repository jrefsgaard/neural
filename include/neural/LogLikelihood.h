#ifndef LOG_LIKELIHOOD_H
#define LOG_LIKELIHOOD_H
#include <armadillo>
#include "CostFunction.h"
#include "Network.h"

namespace neural {

class LogLikelihood : public CostFunction {
  public:
    LogLikelihood() = default;
    ~LogLikelihood() = default;

    using CostFunction::Eval;

    virtual double Eval(arma::vec &a, arma::vec &y);
    virtual arma::vec Delta(arma::vec &a, arma::vec &y, Network &network);
};
}
#endif
