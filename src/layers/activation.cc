#include "activation.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerActivation::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");

            unsigned int activation = 0;
            KASSERT(ReadUnsignedInt(file, &activation),
                    "Failed to read activation type");

            switch (activation) {
            case kLinear:
                activation_type_ = kLinear;
                break;
            case kRelu:
                activation_type_ = kRelu;
                break;
            case kSoftPlus:
                activation_type_ = kSoftPlus;
                break;
            case kHardSigmoid:
                activation_type_ = kHardSigmoid;
                break;
            case kSigmoid:
                activation_type_ = kSigmoid;
                break;
            case kTanh:
                activation_type_ = kTanh;
                break;
            default:
                KASSERT(false, "Unsupported activation type %d", activation);
            }

            return true;
        }
    
        bool KerasLayerActivation::Apply(Tensor* in, Tensor* out) {
            KASSERT(in, "Invalid input");
            KASSERT(out, "Invalid output");

            *out = *in;

            switch (activation_type_) {
            case kLinear:
                break;
            case kRelu:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    if (out->data_[i] < 0.0) {
                        out->data_[i] = 0.0;
                    }
                }
                break;
            case kElu:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    if (out->data_[i] < 0.0) {
                        out->data_[i] = std::expm1(out->data_[i]);
                    }
                }
            case kSoftPlus:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    out->data_[i] = std::log(1.0 + std::exp(out->data_[i]));
                }
                break;
            case kSoftSign:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    out->data_[i] = out->data_[i] / (1.0 + std::abs(out->data_[i]));
                }
                break;
            case kSoftMax:
                if(out->data_.size() > 1){
                    float sum = 0.0;
                    float max = *std::max_element(std::begin(out->data_), std::end(out->data_));
                    for (size_t i = 0; i < out->data_.size(); i++) {
                        out->data_[i] = std::exp(out->data_[i] - max);
                        sum += out->data_[i];
                    }
                    for (size_t i = 0; i < out->data_.size(); i++)
                        out->data_[i] /= sum;
                }
                break;
            case kHardSigmoid:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    float x = (out->data_[i] * 0.2) + 0.5;

                    if (x <= -2.5) {
                        out->data_[i] = 0.0;
                    } else if (x >= 2.5) {
                        out->data_[i] = 1.0;
                    } else {
                        out->data_[i] = x;
                    }
                }
                break;
            case kSigmoid:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    float& x = out->data_[i];

                    if (x >= 0) {
                        out->data_[i] = 1.0 / (1.0 + std::exp(-x));
                    } else {
                        float z = std::exp(x);
                        out->data_[i] = z / (1.0 + z);
                    }
                }
                break;
            case kTanh:
                for (size_t i = 0; i < out->data_.size(); i++) {
                    out->data_[i] = std::tanh(out->data_[i]);
                }
                break;
            default:
                break;
            }

            return true;
        }
    }
}