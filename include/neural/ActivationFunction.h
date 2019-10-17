#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
#include <string>
#include <armadillo>

namespace neural {

class ActivationFunction {
  public:
    ActivationFunction() = default;
    virtual ~ActivationFunction() = default;

    static ActivationFunction * Create(std::string type);

    virtual arma::vec Eval(arma::vec x) = 0;
    virtual arma::vec Deriv(arma::vec x) = 0;
};
}
#endif
