#pragma once

#include "tensor.h"

namespace keras2cpp {
    class BaseLayer {
    public:
        BaseLayer() {}
        virtual ~BaseLayer() {}
        virtual bool LoadLayer(std::ifstream* file) = 0;
        virtual bool Apply(Tensor* in, Tensor* out) = 0;
    };
}
