#include "baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerActivation : public BaseLayer {
            public:
                enum ActivationType {
                    kLinear = 1,
                    kRelu = 2,
                    kSoftPlus = 3,
                    kSigmoid = 4,
                    kTanh = 5,
                    kHardSigmoid = 6,
                    kElu = 7,
                    kSoftSign = 8,
                    kSoftMax = 9
                };
                KerasLayerActivation() : activation_type_(ActivationType::kLinear) {}
                virtual ~KerasLayerActivation() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);
            private:
                ActivationType activation_type_;
        };
    }
}