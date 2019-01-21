#include "flatten.h"
namespace keras2cpp{
    namespace layers{
        bool KerasLayerLSTM::LoadLayer(std::ifstream* file) {
            KASSERT(file, "Invalid file stream");
        
            unsigned int wi_rows = 0;
            KASSERT(ReadUnsignedInt(file, &wi_rows), "Expected Wi rows");
            KASSERT(wi_rows > 0, "Invalid Wi # rows");
        
            unsigned int wi_cols = 0;
            KASSERT(ReadUnsignedInt(file, &wi_cols), "Expected Wi cols");
            KASSERT(wi_cols > 0, "Invalid Wi shape");
        
            unsigned int ui_rows = 0;
            KASSERT(ReadUnsignedInt(file, &ui_rows), "Expected Ui rows");
            KASSERT(ui_rows > 0, "Invalid Ui # rows");
        
            unsigned int ui_cols = 0;
            KASSERT(ReadUnsignedInt(file, &ui_cols), "Expected Ui cols");
            KASSERT(ui_cols > 0, "Invalid Ui shape");
        
            unsigned int bi_shape = 0;
            KASSERT(ReadUnsignedInt(file, &bi_shape), "Expected bi shape");
            KASSERT(bi_shape > 0, "Invalid bi shape");
        
            unsigned int wf_rows = 0;
            KASSERT(ReadUnsignedInt(file, &wf_rows), "Expected Wf rows");
            KASSERT(wf_rows > 0, "Invalid Wf # rows");
        
            unsigned int wf_cols = 0;
            KASSERT(ReadUnsignedInt(file, &wf_cols), "Expected Wf cols");
            KASSERT(wf_cols > 0, "Invalid Wf shape");
        
            unsigned int uf_rows = 0;
            KASSERT(ReadUnsignedInt(file, &uf_rows), "Expected Uf rows");
            KASSERT(uf_rows > 0, "Invalid Uf # rows");
        
            unsigned int uf_cols = 0;
            KASSERT(ReadUnsignedInt(file, &uf_cols), "Expected Uf cols");
            KASSERT(uf_cols > 0, "Invalid Uf shape");
        
            unsigned int bf_shape = 0;
            KASSERT(ReadUnsignedInt(file, &bf_shape), "Expected bf shape");
            KASSERT(bf_shape > 0, "Invalid bf shape");
        
            unsigned int wc_rows = 0;
            KASSERT(ReadUnsignedInt(file, &wc_rows), "Expected Wc rows");
            KASSERT(wc_rows > 0, "Invalid Wc # rows");
        
            unsigned int wc_cols = 0;
            KASSERT(ReadUnsignedInt(file, &wc_cols), "Expected Wc cols");
            KASSERT(wc_cols > 0, "Invalid Wc shape");
        
            unsigned int uc_rows = 0;
            KASSERT(ReadUnsignedInt(file, &uc_rows), "Expected Uc rows");
            KASSERT(uc_rows > 0, "Invalid Uc # rows");
        
            unsigned int uc_cols = 0;
            KASSERT(ReadUnsignedInt(file, &uc_cols), "Expected Uc cols");
            KASSERT(uc_cols > 0, "Invalid Uc shape");
        
            unsigned int bc_shape = 0;
            KASSERT(ReadUnsignedInt(file, &bc_shape), "Expected bc shape");
            KASSERT(bc_shape > 0, "Invalid bc shape");
        
            unsigned int wo_rows = 0;
            KASSERT(ReadUnsignedInt(file, &wo_rows), "Expected Wo rows");
            KASSERT(wo_rows > 0, "Invalid Wo # rows");
        
            unsigned int wo_cols = 0;
            KASSERT(ReadUnsignedInt(file, &wo_cols), "Expected Wo cols");
            KASSERT(wo_cols > 0, "Invalid Wo shape");
        
            unsigned int uo_rows = 0;
            KASSERT(ReadUnsignedInt(file, &uo_rows), "Expected Uo rows");
            KASSERT(uo_rows > 0, "Invalid Uo # rows");
        
            unsigned int uo_cols = 0;
            KASSERT(ReadUnsignedInt(file, &uo_cols), "Expected Uo cols");
            KASSERT(uo_cols > 0, "Invalid Uo shape");
        
            unsigned int bo_shape = 0;
            KASSERT(ReadUnsignedInt(file, &bo_shape), "Expected bo shape");
            KASSERT(bo_shape > 0, "Invalid bo shape");
        
            // Load Input Weights and Biases
            Wi_.Resize(wi_rows, wi_cols);
            KASSERT(ReadFloats(file, Wi_.data_.data(), wi_rows * wi_cols),
                    "Expected Wi weights");
        
            Ui_.Resize(ui_rows, ui_cols);
            KASSERT(ReadFloats(file, Ui_.data_.data(), ui_rows * ui_cols),
                    "Expected Ui weights");
        
            bi_.Resize(1, bi_shape);
            KASSERT(ReadFloats(file, bi_.data_.data(), bi_shape), "Expected bi biases");
        
            // Load Forget Weights and Biases
            Wf_.Resize(wf_rows, wf_cols);
            KASSERT(ReadFloats(file, Wf_.data_.data(), wf_rows * wf_cols),
                    "Expected Wf weights");
        
            Uf_.Resize(uf_rows, uf_cols);
            KASSERT(ReadFloats(file, Uf_.data_.data(), uf_rows * uf_cols),
                    "Expected Uf weights");
        
            bf_.Resize(1, bf_shape);
            KASSERT(ReadFloats(file, bf_.data_.data(), bf_shape), "Expected bf biases");
        
            // Load State Weights and Biases
            Wc_.Resize(wc_rows, wc_cols);
            KASSERT(ReadFloats(file, Wc_.data_.data(), wc_rows * wc_cols),
                    "Expected Wc weights");
        
            Uc_.Resize(uc_rows, uc_cols);
            KASSERT(ReadFloats(file, Uc_.data_.data(), uc_rows * uc_cols),
                    "Expected Uc weights");
        
            bc_.Resize(1, bc_shape);
            KASSERT(ReadFloats(file, bc_.data_.data(), bc_shape), "Expected bc biases");
        
            // Load Output Weights and Biases
            Wo_.Resize(wo_rows, wo_cols);
            KASSERT(ReadFloats(file, Wo_.data_.data(), wo_rows * wo_cols),
                    "Expected Wo weights");
        
            Uo_.Resize(uo_rows, uo_cols);
            KASSERT(ReadFloats(file, Uo_.data_.data(), uo_rows * uo_cols),
                    "Expected Uo weights");
        
            bo_.Resize(1, bo_shape);
            KASSERT(ReadFloats(file, bo_.data_.data(), bo_shape), "Expected bo biases");
        
            KASSERT(innerActivation_.LoadLayer(file),
                    "Failed to load inner activation");
            KASSERT(activation_.LoadLayer(file), "Failed to load activation");
        
            unsigned int return_sequences = 0;
            KASSERT(ReadUnsignedInt(file, &return_sequences),
                    "Expected return_sequences param");
            return_sequences_ = (bool)return_sequences;
        
            return true;
        }
        
        bool KerasLayerLSTM::Apply(Tensor* in, Tensor* out) {
            // Assume bo always keeps the output shape and we will always receive one
            // single sample.
            int outputDim = bo_.dims_[1];
            Tensor ht_1 = Tensor(1, outputDim);
            Tensor ct_1 = Tensor(1, outputDim);
        
            ht_1.Fill(0.0f);
            ct_1.Fill(0.0f);
        
            int steps = in->dims_[0];
        
            Tensor outputs, lastOutput;
        
            if (return_sequences_) {
                outputs.dims_ = {steps, outputDim};
                outputs.data_.reserve(steps * outputDim);
            }
        
            for (int s = 0; s < steps; s++) {
                Tensor x = in->Select(s);
        
                KASSERT(Step(&x, &lastOutput, &ht_1, &ct_1), "Failed to execute step");
        
                if (return_sequences_) {
                    outputs.data_.insert(outputs.data_.end(), lastOutput.data_.begin(),
                                         lastOutput.data_.end());
                }
            }
        
            if (return_sequences_) {
                *out = outputs;
            } else {
                *out = lastOutput;
            }
        
            return true;
        }
        bool KerasLayerLSTM::Step(Tensor* x, Tensor* out, Tensor* ht_1, Tensor* ct_1) {
            Tensor xi = x->Dot(Wi_) + bi_;
            Tensor xf = x->Dot(Wf_) + bf_;
            Tensor xc = x->Dot(Wc_) + bc_;
            Tensor xo = x->Dot(Wo_) + bo_;
        
            Tensor i_ = xi + ht_1->Dot(Ui_);
            Tensor f_ = xf + ht_1->Dot(Uf_);
            Tensor c_ = xc + ht_1->Dot(Uc_);
            Tensor o_ = xo + ht_1->Dot(Uo_);
        
            Tensor i, f, cc, o;
        
            KASSERT(innerActivation_.Apply(&i_, &i),
                    "Failed to apply inner activation on i");
            KASSERT(innerActivation_.Apply(&f_, &f),
                    "Failed to apply inner activation on f");
            KASSERT(activation_.Apply(&c_, &cc), "Failed to apply activation on c_");
            KASSERT(innerActivation_.Apply(&o_, &o),
                    "Failed to apply inner activation on o");
        
            *ct_1 = f.Multiply(*ct_1) + i.Multiply(cc);
        
            KASSERT(activation_.Apply(ct_1, &cc), "Failed to apply activation on c");
            *out = *ht_1 = o.Multiply(cc);
        
            return true;
        }
    }
}