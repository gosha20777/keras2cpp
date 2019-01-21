#include "model.h"
namespace keras2cpp {
    bool KerasModel::LoadModel(const std::string& filename) {
        std::ifstream file(filename.c_str(), std::ios::binary);
        KASSERT(file.is_open(), "Unable to open file %s", filename.c_str());

        unsigned int num_layers = 0;
        KASSERT(ReadUnsignedInt(&file, &num_layers), "Expected number of layers");

        for (unsigned int i = 0; i < num_layers; i++) {
            unsigned int layer_type = 0;
            KASSERT(ReadUnsignedInt(&file, &layer_type), "Expected layer type");

            KerasLayer* layer = NULL;

            switch (layer_type) {
            case kDense:
                layer = new KerasLayerDense();
                break;
            case kConvolution2d:
                layer = new KerasLayerConvolution2d();
                break;
            case kFlatten:
                layer = new KerasLayerFlatten();
                break;
            case kElu:
                layer = new KerasLayerElu();
                break;
            case kActivation:
                layer = new KerasLayerActivation();
                break;
            case kMaxPooling2D:
                layer = new KerasLayerMaxPooling2d();
                break;
            case kLSTM:
                layer = new KerasLayerLSTM();
                break;
            case kEmbedding:
                layer = new KerasLayerEmbedding();
                break;
            case kBatchNormalization:
                layer = new KerasLayerBatchNormalization();
                break;
            default:
                break;
            }

            KASSERT(layer, "Unknown layer type %d", layer_type);

            bool result = layer->LoadLayer(&file);
            if (!result) {
                printf("Failed to load layer %d", i);
                delete layer;
                return false;
            }

            layers_.push_back(layer);
        }

        return true;
    }

    bool KerasModel::Apply(Tensor* in, Tensor* out) {
        Tensor temp_in, temp_out;

        for (unsigned int i = 0; i < layers_.size(); i++) {
            if (i == 0) {
                temp_in = *in;
            }

            KASSERT(layers_[i]->Apply(&temp_in, &temp_out),
                    "Failed to apply layer %d", i);

            temp_in = temp_out;
        }

        *out = temp_out;

        return true;
    }
}