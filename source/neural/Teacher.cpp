#include <neural/Teacher.h>

using namespace std;

namespace neural {

Teacher::Teacher(string cost)
{
  costFunction = unique_ptr<CostFunction>(CostFunction::Create(cost));
}

void Teacher::SetCostFunction(std::string cost)
{
  costFunction = unique_ptr<CostFunction>(CostFunction::Create(cost));
}

CostFunction & Teacher::GetCostFunction()
{
  return *(costFunction.get());
}
}


