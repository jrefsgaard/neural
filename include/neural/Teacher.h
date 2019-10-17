#ifndef TEACHER_H
#define TEACHER_H
#include <memory>
#include <string>
#include "Network.h"
#include "DataSet.h"
#include "CostFunction.h"

namespace neural {

class Teacher {
  protected:
    std::unique_ptr<CostFunction> costFunction;

  public:
    Teacher(std::string cost = "quadratic");
    virtual ~Teacher() = default;

    virtual void Train(Network &network, std::shared_ptr<DataSet> data) = 0;

    void SetCostFunction(std::string cost);

    CostFunction & GetCostFunction();
};
}
#endif
