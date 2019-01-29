#include "../baseLayer.h"
#include "activation.h"
namespace keras2cpp{
    namespace layers{
        class KerasLayerLSTM : public KerasLayer {
            public:
                KerasLayerLSTM() : return_sequences_(false) {}
                virtual ~KerasLayerLSTM() {}
                virtual bool LoadLayer(std::ifstream* file);
                virtual bool Apply(Tensor* in, Tensor* out);
            private:
            bool Step(Tensor* x, Tensor* out, Tensor* ht_1, Tensor* ct_1);
            Tensor Wi_;
            Tensor Ui_;
            Tensor bi_;
            Tensor Wf_;
            Tensor Uf_;
            Tensor bf_;
            Tensor Wc_;
            Tensor Uc_;
            Tensor bc_;
            Tensor Wo_;
            Tensor Uo_;
            Tensor bo_;
            KerasLayerActivation innerActivation_;
            KerasLayerActivation activation_;
            bool return_sequences_;
        };
    }
}