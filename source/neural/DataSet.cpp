#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "neural/DataSet.h"

using namespace std;
using namespace arma;

namespace neural {

DataSet::DataSet(int size)
{
  data.resize(size);
}

DataSet::DataSet(vector<array<Col<double>,2>> rawData)
{
  data = rawData;
}

void DataSet::SetData(vector<array<arma::Col<double>,2>> newData)
{
  data = newData;
}

vector<array<vec,2>> & DataSet::GetData()
{
  return data;
}

array<vec,2> & DataSet::GetDataPoint(int i)
{
  return data.at(i);
}

void DataSet::SetInput(int i, arma::vec input)
{
  data.at(i).at(0) = input;
}

void DataSet::SetOutput(int i, arma::vec output)
{
  data.at(i).at(1) = output;
}

void DataSet::Shuffle()
{
  random_shuffle(data.begin(), data.end() );
}

int DataSet::GetSize()
{
  return data.size();
}

shared_ptr<DataSet> DataSet::GetSubSet(int first, int last)
{
  if(first < 0 || first >= data.size() || last < first || last >= data.size()){
    cout << "DataSet::GetSubSet(): Requested subset out of range!" << endl;
    exit(EXIT_FAILURE);
  }
  vector<array<arma::Col<double>,2>> subset(data.begin() + first, data.begin() + last+1);
  shared_ptr<DataSet> subset_ptr(new DataSet(subset));
  return subset_ptr;
}
}
