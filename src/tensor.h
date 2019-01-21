#include "utils.h"

namespace keras2cpp {
    class Tensor {
    public:
        Tensor() {}

        Tensor(int i) { Resize(i); }

        Tensor(int i, int j) { Resize(i, j); }

        Tensor(int i, int j, int k) { Resize(i, j, k); }

        Tensor(int i, int j, int k, int l) { Resize(i, j, k, l); }

        void Resize(int i) {
            dims_ = {i};
            data_.resize(i);
        }

        void Resize(int i, int j) {
            dims_ = {i, j};
            data_.resize(i * j);
        }

        void Resize(int i, int j, int k) {
            dims_ = {i, j, k};
            data_.resize(i * j * k);
        }

        void Resize(int i, int j, int k, int l) {
            dims_ = {i, j, k, l};
            data_.resize(i * j * k * l);
        }

        inline void Flatten() {
            KDEBUG(dims_.size() > 0, "Invalid tensor");

            int elements = dims_[0];
            for (unsigned int i = 1; i < dims_.size(); i++) {
                elements *= dims_[i];
            }
            dims_ = {elements};
        }

        inline float& operator()(int i) {
            KDEBUG(dims_.size() == 1, "Invalid indexing for tensor");
            KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);

            return data_[i];
        }

        inline float& operator()(int i, int j) {
            KDEBUG(dims_.size() == 2, "Invalid indexing for tensor");
            KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
            KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);

            return data_[dims_[1] * i + j];
        }

        inline float operator()(int i, int j) const {
            KDEBUG(dims_.size() == 2, "Invalid indexing for tensor");
            KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
            KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);

            return data_[dims_[1] * i + j];
        }

        inline float& operator()(int i, int j, int k) {
            KDEBUG(dims_.size() == 3, "Invalid indexing for tensor");
            KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
            KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);
            KDEBUG(k < dims_[2] && k >= 0, "Invalid k: %d (max %d)", k, dims_[2]);

            return data_[dims_[2] * (dims_[1] * i + j) + k];
        }

        inline float& operator()(int i, int j, int k, int l) {
            KDEBUG(dims_.size() == 4, "Invalid indexing for tensor");
            KDEBUG(i < dims_[0] && i >= 0, "Invalid i: %d (max %d)", i, dims_[0]);
            KDEBUG(j < dims_[1] && j >= 0, "Invalid j: %d (max %d)", j, dims_[1]);
            KDEBUG(k < dims_[2] && k >= 0, "Invalid k: %d (max %d)", k, dims_[2]);
            KDEBUG(l < dims_[3] && l >= 0, "Invalid l: %d (max %d)", l, dims_[3]);

            return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
        }

        inline void Fill(float value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        Tensor Unpack(int row) const {
            KASSERT(dims_.size() >= 2, "Invalid tensor");
            std::vector<int> pack_dims =
                std::vector<int>(dims_.begin() + 1, dims_.end());
            int pack_size = std::accumulate(pack_dims.begin(), pack_dims.end(), 0);

            std::vector<float>::const_iterator first =
                data_.begin() + (row * pack_size);
            std::vector<float>::const_iterator last =
                data_.begin() + (row + 1) * pack_size;

            Tensor x = Tensor();
            x.dims_ = pack_dims;
            x.data_ = std::vector<float>(first, last);

            return x;
        }

        Tensor Select(int row) const {
            Tensor x = Unpack(row);
            x.dims_.insert(x.dims_.begin(), 1);

            return x;
        }

        Tensor operator+(const Tensor& other) {
            KASSERT(dims_ == other.dims_,
                    "Cannot add tensors with different dimensions");

            Tensor result;
            result.dims_ = dims_;
            result.data_.reserve(data_.size());

            std::transform(data_.begin(), data_.end(), other.data_.begin(),
                           std::back_inserter(result.data_),
                           [](float x, float y) { return x + y; });

            return result;
        }

        Tensor Multiply(const Tensor& other) {
            KASSERT(dims_ == other.dims_,
                    "Cannot multiply elements with different dimensions");

            Tensor result;
            result.dims_ = dims_;
            result.data_.reserve(data_.size());

            std::transform(data_.begin(), data_.end(), other.data_.begin(),
                           std::back_inserter(result.data_),
                           [](float x, float y) { return x * y; });

            return result;
        }

        Tensor Dot(const Tensor& other) {
            KDEBUG(dims_.size() == 2, "Invalid tensor dimensions");
            KDEBUG(other.dims_.size() == 2, "Invalid tensor dimensions");
            KASSERT(dims_[1] == other.dims_[0],
                    "Cannot multiply with different inner dimensions");

            Tensor tmp(dims_[0], other.dims_[1]);

            for (int i = 0; i < dims_[0]; i++) {
                for (int j = 0; j < other.dims_[1]; j++) {
                    for (int k = 0; k < dims_[1]; k++) {
                        tmp(i, j) += (*this)(i, k) * other(k, j);
                    }
                }
            }

            return tmp;
        }

        Tensor Fma(const Tensor& scale, const Tensor& bias) const noexcept {
            KASSERT(dims_ == scale.dims_, "Invalid tensor dimensions");
            KASSERT(dims_ == bias.dims_, "Invalid tensor dimensions");

            Tensor result;
            result.dims_ = dims_;
            result.data_.resize(data_.size());

            auto k_ = scale.data_.begin();
            auto b_ = bias.data_.begin();
            auto r_ = result.data_.begin();
            for (auto x_ = data_.begin(); x_ != data_.end();)
                *(r_++) = *(x_++) * *(k_++) + *(b_++);
            return result;
        }

        void Print() {
            if (dims_.size() == 1) {
                printf("[ ");
                for (int i = 0; i < dims_[0]; i++) {
                    printf("%f ", (*this)(i));
                }
                printf("]\n");
            } else if (dims_.size() == 2) {
                printf("[\n");
                for (int i = 0; i < dims_[0]; i++) {
                    printf(" [ ");
                    for (int j = 0; j < dims_[1]; j++) {
                        printf("%f ", (*this)(i, j));
                    }
                    printf("]\n");
                }
                printf("]\n");
            } else if (dims_.size() == 3) {
                printf("[\n");
                for (int i = 0; i < dims_[0]; i++) {
                    printf(" [\n");
                    for (int j = 0; j < dims_[1]; j++) {
                        printf("  [ ");
                        for (int k = 0; k < dims_[2]; k++) {
                            printf("%f ", (*this)(i, j, k));
                        }
                        printf("  ]\n");
                    }
                    printf(" ]\n");
                }
                printf("]\n");
            } else if (dims_.size() == 4) {
                printf("[\n");
                for (int i = 0; i < dims_[0]; i++) {
                    printf(" [\n");
                    for (int j = 0; j < dims_[1]; j++) {
                        printf("  [\n");
                        for (int k = 0; k < dims_[2]; k++) {
                            printf("   [");
                            for (int l = 0; l < dims_[3]; l++) {
                                printf("%f ", (*this)(i, j, k, l));
                            }
                            printf("]\n");
                        }
                        printf("  ]\n");
                    }
                    printf(" ]\n");
                }
                printf("]\n");
            }
        }

        void PrintShape() {
            printf("(");
            for (unsigned int i = 0; i < dims_.size(); i++) {
                printf("%d ", dims_[i]);
            }
            printf(")\n");
        }

        std::vector<int> dims_;
        std::vector<float> data_;
    };
}