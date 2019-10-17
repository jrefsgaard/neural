#ifndef RECTIFIED_LINEAR_H
#define RECTIFIED_LINEAR_H
#include "ActivationFunction.h"
#include <armadillo>

namespace neural {

class RectifiedLinear : public ActivationFunction {
  public:
    RectifiedLinear() = default;
    ~RectifiedLinear() = default;

    virtual arma::vec Eval(arma::vec x);
    virtual arma::vec Deriv(arma::vec x);
};
}
#endif
