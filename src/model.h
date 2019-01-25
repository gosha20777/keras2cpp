#pragma once
#include "baseLayer.h"
namespace keras2cpp {
    class Model final : public Layer<Model> {
        std::vector<std::unique_ptr<BaseLayer>> layers_;

    public:
        Model(Stream& file);
        Tensor operator()(const Tensor& in) const noexcept override;
    };
}