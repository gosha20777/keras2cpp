#pragma once
#include "baseLayer.h"
namespace keras2cpp {
    class Model : public Layer<Model> {
        enum _LayerType : unsigned {
            Dense = 1,
            Conv1D = 2,
            Conv2D = 3,
            LocallyConnected1D = 4,
            LocallyConnected2D = 5,
            Flatten = 6,
            ELU = 7,
            Activation = 8,
            MaxPooling2D = 9,
            LSTM = 10,
            Embedding = 11,
            BatchNormalization = 12,
        };
        std::vector<std::unique_ptr<BaseLayer>> layers_;
        
        static std::unique_ptr<BaseLayer> make_layer(Stream&);

    public:
        Model(Stream& file);
        Tensor operator()(const Tensor& in) const noexcept override;
    };
}