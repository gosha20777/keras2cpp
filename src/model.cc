#include "model.h"
#include "layers/conv1d.h"
#include "layers/conv2d.h"
#include "layers/dense.h"
#include "layers/elu.h"
#include "layers/embedding.h"
#include "layers/flatten.h"
#include "layers/locally1d.h"
#include "layers/locally2d.h"
#include "layers/lstm.h"
#include "layers/maxpooling2d.h"
#include "layers/normalization.h"

namespace keras2cpp {
    namespace layers {
        using types = std::tuple<
            Dense,               //0
            Conv1D,              //1
            Conv2D,              //2
            LocallyConnected1D,  //3
            LocallyConnected2D,  //4
            Flatten,             //5
            ELU,                 //6
            Activation,          //7
            MaxPooling2D,        //8
            LSTM,                //9
            Embedding,           //10
            BatchNormalization>; //11
    } 

    template <size_t... I>
    std::unique_ptr<BaseLayer>
    _make_layer(std::index_sequence<I...>, Stream& file) {
        auto id = static_cast<unsigned>(file);
        std::unique_ptr<BaseLayer> layer;
        bool found = (... || [&]() {
            if (id != I)
                return false;
            layer = std::move(
                std::make_unique<
                    std::decay_t<std::tuple_element_t<I, layers::types>>>(file));
            return true;
        }());

        if (!found)
            throw std::domain_error("Layer not implemented");
        return layer;
    }

    Model::Model(Stream& file) {
        auto count = static_cast<unsigned>(file);
        layers_.reserve(count);
        for (size_t i = 0; i != count; ++i)
            layers_.push_back(_make_layer(
                std::make_index_sequence<std::tuple_size_v<layers::types>>(),
                file));
    }

    Tensor Model::operator()(const Tensor& in) const noexcept {
        Tensor out = in;
        for (auto&& layer : layers_)
            out = (*layer)(out);
        return out;
    }
}
