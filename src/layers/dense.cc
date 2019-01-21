#include "dense.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerDense::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");

            unsigned int weights_rows = 0;
            KASSERT(ReadUnsignedInt(file, &weights_rows), "Expected weight rows");
            KASSERT(weights_rows > 0, "Invalid weights # rows");

            unsigned int weights_cols = 0;
            KASSERT(ReadUnsignedInt(file, &weights_cols), "Expected weight cols");
            KASSERT(weights_cols > 0, "Invalid weights shape");

            unsigned int biases_shape = 0;
            KASSERT(ReadUnsignedInt(file, &biases_shape), "Expected biases shape");
            KASSERT(biases_shape > 0, "Invalid biases shape");

            weights_.Resize(weights_rows, weights_cols);
            KASSERT(
                ReadFloats(file, weights_.data_.data(), weights_rows * weights_cols),
                "Expected weights");

            biases_.Resize(biases_shape);
            KASSERT(ReadFloats(file, biases_.data_.data(), biases_shape),
                    "Expected biases");

            KASSERT(activation_.LoadLayer(file), "Failed to load activation");

            return true;
        }

        bool KerasLayerDense::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");
            KASSERT(in->dims_.size() <= 2, "Invalid input dimensions");

            if (in->dims_.size() == 2) {
                KASSERT(in->dims_[1] == weights_.dims_[0], "Dimension mismatch %d %d",
                        in->dims_[1], weights_.dims_[0]);
            }

            Tensor tmp(weights_.dims_[1]);

            for (int i = 0; i < weights_.dims_[0]; i++) {
                for (int j = 0; j < weights_.dims_[1]; j++) {
                    tmp(j) += (*in)(i)*weights_(i, j);
                }
            }

            for (int i = 0; i < biases_.dims_[0]; i++) {
                tmp(i) += biases_(i);
            }

            KASSERT(activation_.Apply(&tmp, out), "Failed to apply activation");

            return true;
        }
    }
}