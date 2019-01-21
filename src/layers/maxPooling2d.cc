#include "maxPooling2d.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerMaxPooling2d::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
        
            KASSERT(ReadUnsignedInt(file, &pool_size_j_), "Expected pool size j");
            KASSERT(ReadUnsignedInt(file, &pool_size_k_), "Expected pool size k");
        
            return true;
        }
        
        bool KerasLayerMaxPooling2d::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");
        
            KASSERT(in->dims_.size() == 3, "Input must have 3 dimensions");
        
            Tensor tmp(in->dims_[0], in->dims_[1] / pool_size_j_,
                       in->dims_[2] / pool_size_k_);
        
            for (int i = 0; i < tmp.dims_[0]; i++) {
                for (int j = 0; j < tmp.dims_[1]; j++) {
                    const int tj = j * pool_size_j_;
        
                    for (int k = 0; k < tmp.dims_[2]; k++) {
                        const int tk = k * pool_size_k_;
        
                        // Find maximum value over patch starting at tj, tk.
                        float max_val = -std::numeric_limits<float>::infinity();
        
                        for (unsigned int pj = 0; pj < pool_size_j_; pj++) {
                            for (unsigned int pk = 0; pk < pool_size_k_; pk++) {
                                const float& pool_val = (*in)(i, tj + pj, tk + pk);
                                if (pool_val > max_val) {
                                    max_val = pool_val;
                                }
                            }
                        }
        
                        tmp(i, j, k) = max_val;
                    }
                }
            }
        
            *out = tmp;
        
            return true;
        }
    }
}