#ifndef TANH_FUNCTION_H
#define TANH_FUNCTION_H
#include "ActivationFunction.h"
#include <armadillo>

namespace neural {

class TanhFunction : public ActivationFunction {
  public:
    TanhFunction() = default;
    ~TanhFunction() = default;

    virtual arma::vec Eval(arma::vec x);
    virtual arma::vec Deriv(arma::vec x);
};
}
#endif
