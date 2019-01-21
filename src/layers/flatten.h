#include "baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerFlatten : public KerasLayer {
            public:
                KerasLayerFlatten() {}
                virtual ~KerasLayerFlatten() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);

            private:
        };
    }
}