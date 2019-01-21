#include "baseLayer.h"
#include "activation.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerDense : public KerasLayer {
            public:
                KerasLayerDense() {}
                virtual ~KerasLayerDense() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);

            private:
                Tensor weights_;
                Tensor biases_;
                KerasLayerActivation activation_;
        };
    }
}