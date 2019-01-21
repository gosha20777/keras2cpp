#include "baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerBatchNormalization : public KerasLayer {
        public:
            KerasLayerBatchNormalization() {}
            virtual ~KerasLayerBatchNormalization() {}
            virtual bool LoadLayer(std::ifstream* file);
            virtual bool Apply(Tensor* in, Tensor* out);
        private:
            Tensor weights_;
            Tensor biases_;
        };
    }
}