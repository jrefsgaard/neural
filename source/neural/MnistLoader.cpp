#include "neural/MnistLoader.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace arma;

namespace neural {

void MnistLoader::Open(string imageFileName, string labelFileName)
{
  //Following is taken from StackOverflow
  //https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c/10409376#10409376
  auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };

  typedef unsigned char uchar;

  //Reading images...
  ifstream imageFile(imageFileName, ios::binary);
  if(imageFile.is_open()) {
    int magic_number = 0, n_rows = 0, n_cols = 0;
    int number_of_images = 0, image_size = 0;

    imageFile.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2051){
      throw runtime_error("MnistLoader::Open(): Invalid MNIST image file!");
    }

    imageFile.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
    imageFile.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    imageFile.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

    image_size = n_rows * n_cols;

    //uchar** _dataset = new uchar*[number_of_images];
    data = shared_ptr<DataSet>(new DataSet(number_of_images));
    uchar *dataI = new uchar[image_size];
    for(int i = 0; i<number_of_images; i++) {
      //_dataset[i] = new uchar[image_size];
      imageFile.read((char *)dataI, image_size);
      vec vi; vi.set_size(image_size);
      for(int j=0; j<image_size; j++){
        vi(j) = (double)(dataI[j])/255;
      }
      data->SetInput(i,vi);
    }
    delete [] dataI;
  }
  else {
    throw runtime_error("MnistLoader::Open(): Cannot open file `" + imageFileName + "`!");
  }

  //Reading labels...
  ifstream labelFile(labelFileName, ios::binary);

  if(labelFile.is_open()) {
    int magic_number = 0;
    int number_of_labels = 0;
    labelFile.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2049) throw runtime_error("MnistLoader::Open(): Invalid MNIST label file!");

    labelFile.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

    //uchar* _dataset = new uchar[number_of_labels];
    for(int i = 0; i < number_of_labels; i++) {
      uchar label;
      vec vi; vi.zeros(10);
      labelFile.read((char*)&label, 1);
      vi((int)label) = 1;
      data->SetOutput(i,vi);
    }
  }
  else {
    throw runtime_error("MnistLoader()::Open(): Unable to open file `" + labelFileName + "`!");
  }
  return;
}

shared_ptr<DataSet> MnistLoader::GetData()
{
  return data;
}
}
