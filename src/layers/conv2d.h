#include "baseLayer.h"
#include "activation.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerConvolution2d : public KerasLayer {
            public:
                KerasLayerConvolution2d() {}
                virtual ~KerasLayerConvolution2d() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);
            private:
                Tensor weights_;
                Tensor biases_;
                KerasLayerActivation activation_;
        };
    }
}