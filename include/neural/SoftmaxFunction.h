#ifndef SOFTMAX_FUNCTION_H
#define SOFTMAX_FUNCTION_H
#include "ActivationFunction.h"
#include <armadillo>

namespace neural {

class SoftmaxFunction : public ActivationFunction {
  public:
    SoftmaxFunction() = default;
    ~SoftmaxFunction() = default;

    virtual arma::vec Eval(arma::vec x);

    /**
    * This function should never be used (seriously).
    */
    virtual arma::vec Deriv(arma::vec x);
};
}
#endif
