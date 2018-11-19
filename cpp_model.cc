#include "keras_model.h"

int main() 
{
    // Initialize model.
    KerasModel model;
    model.LoadModel("example.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in(10);
    in.data_ = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

    // Run prediction.
    Tensor out;
    model.Apply(&in, &out);
    out.Print();
    return 0;
}