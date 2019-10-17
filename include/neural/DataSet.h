#ifndef DATA_SET_H
#define DATA_SET_H
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <armadillo>

namespace neural {

class DataSet {
  private:
    //The first vector is the input and the second is the desired output.
    std::vector<std::array<arma::Col<double>,2>> data;

  public:
    DataSet(int size);
    DataSet(std::vector<std::array<arma::Col<double>,2>> rawData);
    ~DataSet() = default;

    void SetData(std::vector<std::array<arma::Col<double>,2>> newData);
    std::vector<std::array<arma::Col<double>,2>> & GetData();
    std::array<arma::Col<double>,2> & GetDataPoint(int i);

    void SetInput(int i, arma::Col<double> input);
    void SetOutput(int i, arma::Col<double> output);

    void Shuffle();

    int GetSize();

    std::shared_ptr<DataSet> GetSubSet(int first, int last);
};
}
#endif
