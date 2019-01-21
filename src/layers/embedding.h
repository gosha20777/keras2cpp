#include "baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerEmbedding : public KerasLayer {
            public:
                KerasLayerEmbedding() {}
                virtual ~KerasLayerEmbedding() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);
            private:
                Tensor weights_;
        };
    }
}