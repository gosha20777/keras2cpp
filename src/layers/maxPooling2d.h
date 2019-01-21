#include "baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerMaxPooling2d : public KerasLayer {
            public:
                KerasLayerMaxPooling2d() : pool_size_j_(0), pool_size_k_(0) {}
                virtual ~KerasLayerMaxPooling2d() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);
            private:
                unsigned int pool_size_j_;
                unsigned int pool_size_k_;
        };
    }
}