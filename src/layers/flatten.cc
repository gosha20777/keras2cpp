#include "flatten.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerFlatten::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
            return true;
        }
        bool KerasLayerFlatten::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");
            *out = *in;
            out->Flatten();
            return true;
        }
    }
}