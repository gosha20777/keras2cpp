#include "conv2d.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerConvolution2d::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
        
            unsigned int weights_i = 0;
            KASSERT(ReadUnsignedInt(file, &weights_i), "Expected weights_i");
            KASSERT(weights_i > 0, "Invalid weights # i");
        
            unsigned int weights_j = 0;
            KASSERT(ReadUnsignedInt(file, &weights_j), "Expected weights_j");
            KASSERT(weights_j > 0, "Invalid weights # j");
        
            unsigned int weights_k = 0;
            KASSERT(ReadUnsignedInt(file, &weights_k), "Expected weights_k");
            KASSERT(weights_k > 0, "Invalid weights # k");
        
            unsigned int weights_l = 0;
            KASSERT(ReadUnsignedInt(file, &weights_l), "Expected weights_l");
            KASSERT(weights_l > 0, "Invalid weights # l");
        
            unsigned int biases_shape = 0;
            KASSERT(ReadUnsignedInt(file, &biases_shape), "Expected biases shape");
            KASSERT(biases_shape > 0, "Invalid biases shape");
        
            weights_.Resize(weights_i, weights_j, weights_k, weights_l);
            KASSERT(ReadFloats(file, weights_.data_.data(),
                               weights_i * weights_j * weights_k * weights_l),
                    "Expected weights");
        
            biases_.Resize(biases_shape);
            KASSERT(ReadFloats(file, biases_.data_.data(), biases_shape),
                    "Expected biases");
        
            KASSERT(activation_.LoadLayer(file), "Failed to load activation");
        
            return true;
        }
        
        bool KerasLayerConvolution2d::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");
        
            KASSERT(in->dims_[0] == weights_.dims_[1],
                    "Input 'depth' doesn't match kernel 'depth'");
        
            int st_nj = (weights_.dims_[2] - 1) / 2;
            int st_pj = (weights_.dims_[2]) / 2;
            int st_nk = (weights_.dims_[3] - 1) / 2;
            int st_pk = (weights_.dims_[3]) / 2;
        
            Tensor tmp(weights_.dims_[0], in->dims_[1] - st_nj - st_pj,
                       in->dims_[2] - st_nk - st_pk);
        
            // Iterate over each kernel.
            for (int i = 0; i < weights_.dims_[0]; i++) {
                // Iterate over each 'depth'.
                for (int j = 0; j < weights_.dims_[1]; j++) {
                    // 2D convolution in x and y (k and l in Tensor dimensions).
                    for (int tj = st_nj; tj < in->dims_[1] - st_pj; tj++) {
                        for (int tk = st_nk; tk < in->dims_[2] - st_pk; tk++) {
                            // Iterate over kernel.
                            for (int k = 0; k < weights_.dims_[2]; k++) {
                                for (int l = 0; l < weights_.dims_[3]; l++) {
                                    const float& weight = weights_(i, j, k, l);
                                    const float& value =
                                        (*in)(j, tj - st_nj + k, tk - st_nk + l);
        
                                    tmp(i, tj - st_nj, tk - st_nk) += weight * value;
                                }
                            }
                        }
                    }
                }
        
                // Apply kernel bias to all points in output.
                for (int j = 0; j < tmp.dims_[1]; j++) {
                    for (int k = 0; k < tmp.dims_[2]; k++) {
                        tmp(i, j, k) += biases_(i);
                    }
                }
            }
        
            KASSERT(activation_.Apply(&tmp, out), "Failed to apply activation");
        
            return true;
        }
    }
}