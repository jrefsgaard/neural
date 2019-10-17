#ifndef LOADER_H
#define LOADER_H
#include "DataSet.h"
#include <memory>

namespace neural {

class Loader {
  public:
    Loader() = default;
    ~Loader() = default;

    virtual std::shared_ptr<DataSet> GetData() = 0;
};
}
#endif
