#include "baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerElu : public KerasLayer {
            public:
                KerasLayerElu() : alpha_(1.0f) {}
                virtual ~KerasLayerElu() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);
            private:
                float alpha_;
        };
    }
}