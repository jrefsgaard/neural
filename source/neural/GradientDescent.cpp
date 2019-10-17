#include <array>
#include <vector>
#include <armadillo>
#include <cstdlib>
#include <neural/GradientDescent.h>
#include <neural/QuadraticCost.h>

using namespace std;
using namespace arma;

namespace neural {

void GradientDescent::CompleteEpoch(Network &network, shared_ptr<DataSet> data)
{
  int L = network.L(); 
  int N = data->GetSize();
  vector<array<vec,2>> &data_vec = data->GetData();

  //Prepare container for the incremental weights.
  vector<mat> dw; dw.resize(L);
  vector<vec> db; db.resize(L);
  for(int l=1; l<L; l++){
    dw.at(l).zeros(size(network.GetLayer(l).GetWeights()));
    db.at(l).zeros(size(network.GetLayer(l).GetBiases()));
  }

  for(int i=0; i<data_vec.size(); i++){
    //1: Retrieve data point.
    array<vec,2> &point = data_vec.at(i);
    vec &x = point[0];
    vec &y = point[1];

    //2: Feedforward.
    vec output = network.FeedForward(x);
    if(output.has_nan()) exit(EXIT_FAILURE);
    vector<vec> deltas; deltas.resize(L);

    //3: Calculate delta(L).
    //QuadraticCost C;
    //vec nabla = C.Nabla(output,y);
    //cout << "nabla_C = " << nabla << endl;
    //vec sigmap = network.GetActivationFunction().Deriv(network.Z(L-1));
    //deltas.at(L-1) = nabla % sigmap;
    deltas.at(L-1) = costFunction->Delta(output,y,network);
    if(deltas.at(L-1).has_nan()) exit(EXIT_FAILURE);
    //cout << "delta_L = " << deltas.at(L-1) << endl;

    //4: We backpropagate to find delta(l).
    for(int l=L-2; l>0; l--){
      const mat &w = network.GetLayer(l+1).GetWeights();
      vec sigmap = network.GetActivationFunction(l).Deriv(network.Z(l));
      deltas.at(l) = (w.t() * deltas.at(l+1)) % sigmap;
      //cout << "delta_" << l << " = " << deltas.at(l) << endl;
    }

    //5: The derivatives are calculated.
    for(int l=1; l<L; l++){
      dw.at(l) += deltas.at(l) * network.A(l-1).t();
      db.at(l) += deltas.at(l);
    }
    
    //Check if a mini-batch has been completed.
    if((i+1) % miniBatchSize == 0){
      for(int l=1; l<L; l++){
        const mat &wl = network.GetLayer(l).GetWeights();
        mat dwl = -learningRate*(regularisationParameter/N * wl + 1./miniBatchSize * dw.at(l));
        //cout << "Old weight = " << network.GetLayer(l).GetWeights() << endl;
        //cout << "  dw = " << dwl << endl;
        network.GetLayer(l).IncrementWeights(dwl);
        dw.at(l).zeros();
        vec dbl = -learningRate/miniBatchSize * db.at(l);
        //cout << "Old bias = " << network.GetLayer(l).GetBiases() << endl;
        //cout << "  db = " << dbl << endl;
        network.GetLayer(l).IncrementBiases(dbl);
        db.at(l).zeros();
      }
    }
  }
}

void GradientDescent::Train(Network &network, shared_ptr<DataSet> data)
{
  for(int i=0; i<epochs; i++){
    //QuadraticCost cost;
    cout << "Epoch " << i << ":  cost = " << costFunction->Eval(network,data) << endl;
    data->Shuffle();
    CompleteEpoch(network,data);
  }
}

void GradientDescent::SetLearningRate(double learningrate)
{
  learningRate = learningrate;
}

void GradientDescent::SetNEpochs(int n_epochs)
{
  epochs = n_epochs;
}

void GradientDescent::SetMiniBatchSize(int minibatchsize)
{
  miniBatchSize = minibatchsize;
}

void GradientDescent::SetRegularisationParameter(double regularisation_parameter)
{
  regularisationParameter = regularisation_parameter;
}

double GradientDescent::GetLearningRate()
{
  return learningRate;
}

int GradientDescent::GetNEpochs()
{
  return epochs;
}

int GradientDescent::GetMiniBatchSize()
{
  return miniBatchSize;
}

double GradientDescent::GetRegularisationParameter()
{
  return regularisationParameter;
}
}
