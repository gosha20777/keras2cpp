#include "src/model.h"

using keras2cpp::Model;
using keras2cpp::Tensor;

int main() {
    // Initialize model.
    auto model = Model::load("example.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in{10};
    in.data_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Run prediction.
    Tensor out = model(in);
    out.print();
    return 0;
}
