#include "elu.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerElu::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
        
            KASSERT(ReadFloat(file, &alpha_), "Failed to read alpha");
        
            return true;
        }
        
        bool KerasLayerElu::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");
        
            *out = *in;
        
            for (size_t i = 0; i < out->data_.size(); i++) {
                if (out->data_[i] < 0.0) {
                    out->data_[i] = alpha_ * (exp(out->data_[i]) - 1.0);
                }
            }
        
            return true;
        }
    }
}