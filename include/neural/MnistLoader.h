#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include "Loader.h"
#include "DataSet.h"
#include <string>
#include <memory>

namespace neural {

class MnistLoader : public Loader {
  private:
    std::shared_ptr<DataSet> data;

  public:
    MnistLoader() = default;
    ~MnistLoader() = default;

    void Open(std::string imageFile, std::string labelFile);

    virtual std::shared_ptr<DataSet> GetData();
};
}
#endif
