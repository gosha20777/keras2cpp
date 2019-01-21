#include "baseLayer.h"
namespace keras2cpp {
    class KerasModel {
        public:
            enum LayerType {
                kDense = 1,
                kConvolution2d = 2,
                kFlatten = 3,
                kElu = 4,
                kActivation = 5,
                kMaxPooling2D = 6,
                kLSTM = 7,
                kEmbedding = 8,
                kBatchNormalization = 9
            };
            KerasModel() {}
            virtual ~KerasModel() {
                for (unsigned int i = 0; i < layers_.size(); i++) {
                    delete layers_[i];
                }              
            }
            virtual bool LoadModel(const std::string& filename);
            virtual bool Apply(Tensor* in, Tensor* out);
        private:
            std::vector<KerasLayer*> layers_;
    };
}