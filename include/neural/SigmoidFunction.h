#ifndef SIGMOID_FUNCTION_H
#define SIGMOID_FUNCTION_H
#include "ActivationFunction.h"
#include <armadillo>

namespace neural {

class SigmoidFunction : public ActivationFunction {
  public:
    SigmoidFunction() = default;
    ~SigmoidFunction() = default;

    virtual arma::vec Eval(arma::vec x);
    virtual arma::vec Deriv(arma::vec x);
};
}
#endif
