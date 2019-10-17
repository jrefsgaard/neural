#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H
#include <memory>
#include "Teacher.h"
#include "Network.h"
#include "DataSet.h"

namespace neural {

class GradientDescent : public Teacher {
  private:
    double learningRate;
    int miniBatchSize;
    int epochs;
    double regularisationParameter;

    void CompleteEpoch(Network &network, std::shared_ptr<DataSet> data);

  public:
    GradientDescent(double learningrate = 1., int n_epochs = 10, int minibatchsize = 10, double regularisation_parameter = 0.)
      : learningRate(learningrate), epochs(n_epochs), miniBatchSize(minibatchsize), regularisationParameter(regularisation_parameter) {};
    ~GradientDescent() = default;

    virtual void Train(Network &network, std::shared_ptr<DataSet> data);

    void SetLearningRate(double learningrate);
    void SetNEpochs(int n_epochs);
    void SetMiniBatchSize(int minibatchsize);
    void SetRegularisationParameter(double regularisation_parameter);

    double GetLearningRate();
    int GetNEpochs();
    int GetMiniBatchSize();
    double GetRegularisationParameter();
};
}
#endif
