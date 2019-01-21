#include "batchNormalization.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerBatchNormalization::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
            unsigned int weights_rows = 0;
            KASSERT(ReadUnsignedInt(file, &weights_rows), "Expected weight rows");
            KASSERT(weights_rows > 0, "Invalid weights # rows");
            unsigned int biases_shape = 0;
            KASSERT(ReadUnsignedInt(file, &biases_shape), "Expected biases shape");
            KASSERT(biases_shape > 0, "Invalid biases shape");

            weights_.Resize(weights_rows);
            KASSERT(
                ReadFloats(file, weights_.data_.data(), weights_rows),
                "Expected weights");

            biases_.Resize(biases_shape);
            KASSERT(ReadFloats(file, biases_.data_.data(), biases_shape),
                    "Expected biases");
            return true;
        }

        bool KerasLayerBatchNormalization::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");
            *out = *in;
            out->Fma(weights_, biases_);
            return true;
        }
    }
}