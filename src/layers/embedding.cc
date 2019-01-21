#include "embedding.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerEmbedding::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
        
            unsigned int weights_rows = 0;
            KASSERT(ReadUnsignedInt(file, &weights_rows), "Expected weight rows");
            KASSERT(weights_rows > 0, "Invalid weights # rows");
        
            unsigned int weights_cols = 0;
            KASSERT(ReadUnsignedInt(file, &weights_cols), "Expected weight cols");
            KASSERT(weights_cols > 0, "Invalid weights shape");
        
            weights_.Resize(weights_rows, weights_cols);
            KASSERT(
                ReadFloats(file, weights_.data_.data(), weights_rows * weights_cols),
                "Expected weights");
        
            return true;
        }
        
        bool KerasLayerEmbedding::Apply(Tensor* in, Tensor* out) {
            int output_rows = in->dims_[1];
            int output_cols = weights_.dims_[1];
            out->dims_ = {output_rows, output_cols};
            out->data_.reserve(output_rows * output_cols);
        
            std::for_each(in->data_.begin(), in->data_.end(), [=](float i) {
                std::vector<float>::const_iterator first =
                    this->weights_.data_.begin() + (i * output_cols);
                std::vector<float>::const_iterator last =
                    this->weights_.data_.begin() + (i + 1) * output_cols;
        
                out->data_.insert(out->data_.end(), first, last);
            });
        
            return true;
        }
    }
}